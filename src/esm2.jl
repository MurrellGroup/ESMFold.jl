struct ESM2Config
    num_layers::Int
    embed_dim::Int
    attention_heads::Int
    token_dropout::Bool
    alphabet::Alphabet
end

struct ESM2Output
    logits::AbstractArray
    representations::Dict{Int,<:AbstractArray}
    attentions
end

@concrete struct RobertaLMHead <: Onion.Layer
    dense
    layer_norm
    weight
    bias
end

@layer RobertaLMHead

function RobertaLMHead(embed_dim::Int, output_dim::Int, weight)
    dense = Onion.Linear(embed_dim => embed_dim)
    layer_norm = Onion.LayerNorm(embed_dim; eps=1f-5)
    bias = zeros(Float32, output_dim)
    return RobertaLMHead(dense, layer_norm, weight, bias)
end

function gelu(x)
    # Use NNlib's fused GELU for better GPU performance
    return NNlib.gelu.(x)
end

@concrete struct TransformerLayer <: Onion.Layer
    self_attn
    self_attn_layer_norm
    fc1
    fc2
    final_layer_norm
end

@layer TransformerLayer

function TransformerLayer(
    embed_dim::Int,
    ffn_embed_dim::Int,
    attention_heads::Int;
    use_rotary_embeddings::Bool = true,
)
    self_attn = ESMMultiheadAttention(
        embed_dim,
        attention_heads;
        use_rotary_embeddings = use_rotary_embeddings,
    )
    self_attn_layer_norm = Onion.LayerNorm(embed_dim; eps=1f-5)
    fc1 = Onion.Linear(embed_dim => ffn_embed_dim)
    fc2 = Onion.Linear(ffn_embed_dim => embed_dim)
    final_layer_norm = Onion.LayerNorm(embed_dim; eps=1f-5)
    return TransformerLayer(self_attn, self_attn_layer_norm, fc1, fc2, final_layer_norm)
end

function (layer::TransformerLayer)(
    x;
    self_attn_padding_mask = nothing,
    need_head_weights::Bool = false,
    _precomputed_attn_bias = nothing,
)
    residual = x
    x = layer.self_attn_layer_norm(x)
    attn_out, attn_weights = layer.self_attn(
        x;
        key_padding_mask = self_attn_padding_mask,
        need_head_weights = need_head_weights,
        _precomputed_attn_bias = _precomputed_attn_bias,
    )

    # In-place residual additions on CUDA to avoid allocating new arrays
    if isa(residual, CuArray)
        residual .+= attn_out
        x = residual

        residual = x
        x = layer.final_layer_norm(x)
        x = layer.fc1(x)
        @. x = NNlib.gelu(x)  # in-place GELU
        x = layer.fc2(x)
        residual .+= x
        return residual, attn_weights
    end

    x = residual .+ attn_out

    residual = x
    x = layer.final_layer_norm(x)
    x = gelu(layer.fc1(x))
    x = layer.fc2(x)
    x = residual .+ x

    return x, attn_weights
end

@concrete struct ESM2 <: Onion.Layer
    num_layers::Int
    embed_dim::Int
    attention_heads::Int
    alphabet::Alphabet
    token_dropout::Bool
    embed_scale::Float32
    embed_tokens
    layers
    emb_layer_norm_after
    lm_head
end

@layer ESM2

function ESM2(
    num_layers::Int = 33,
    embed_dim::Int = 1280,
    attention_heads::Int = 20;
    alphabet::Alphabet = Alphabet_from_architecture("ESM-1b"),
    token_dropout::Bool = true,
)
    embed_tokens = Flux.Embedding(length(alphabet.all_toks), embed_dim)
    layers = [
        TransformerLayer(
            embed_dim,
            4 * embed_dim,
            attention_heads;
            use_rotary_embeddings = true,
        )
        for _ in 1:num_layers
    ]
    emb_layer_norm_after = Onion.LayerNorm(embed_dim; eps=1f-5)
    lm_head = RobertaLMHead(embed_dim, length(alphabet.all_toks), embed_tokens.weight)

    return ESM2(
        num_layers,
        embed_dim,
        attention_heads,
        alphabet,
        token_dropout,
        1f0,
        embed_tokens,
        layers,
        emb_layer_norm_after,
        lm_head,
    )
end

function (lm::RobertaLMHead)(features)
    x = lm.dense(features)
    x = gelu(x)
    x = lm.layer_norm(x)
    # features are (C, T, B); weight is (V, C)
    x = reshape(x, size(x, 1), :)
    y = lm.weight' * x .+ lm.bias
    return reshape(y, size(lm.bias, 1), size(features, 2), size(features, 3))
end

function (model::ESM2)(
    tokens::AbstractArray{Int,2};
    repr_layers = Int[],
    need_head_weights::Bool = false,
    return_contacts::Bool = false,
)
    return_contacts && (need_head_weights = true)

    # tokens: (B, T) with 0-based indices
    padding_mask = tokens .== model.alphabet.padding_idx

    tokens_tb = permutedims(tokens, (2, 1))
    x = model.embed_scale .* model.embed_tokens(tokens_tb .+ 1)

    if model.token_dropout
        mask_positions = tokens .== model.alphabet.mask_idx
        mask_positions_tb = permutedims(mask_positions, (2, 1))
        x .= x .* reshape(1 .- mask_positions_tb, 1, size(x, 2), size(x, 3))

        mask_ratio_train = 0.15f0 * 0.8f0
        src_lengths = sum(.!padding_mask, dims=2)
        mask_ratio_observed = Float32.(sum(mask_positions, dims=2)) ./ Float32.(src_lengths)
        scale = (1f0 - mask_ratio_train) ./ (1f0 .- mask_ratio_observed)
        scale = reshape(scale, 1, 1, size(scale, 1))
        x .= x .* scale
    end

    padding_mask_tb = permutedims(padding_mask, (2, 1))
    x .= x .* reshape(1 .- padding_mask_tb, 1, size(x, 2), size(x, 3))

    repr_set = Set(repr_layers)
    hidden_representations = Dict{Int,AbstractArray}()
    if 0 in repr_set
        hidden_representations[0] = permutedims(x, (3, 2, 1))
    end

    attn_weights_all = need_head_weights ? Vector{AbstractArray}(undef, model.num_layers) : nothing

    padding_mask_for_attn = any(padding_mask) ? padding_mask : nothing

    # Pre-compute the attention bias once for all 33 layers (saves ~130 kernel launches)
    precomputed_attn_bias = nothing
    if padding_mask_for_attn !== nothing && isa(x, CuArray)
        head_dim = model.embed_dim ÷ model.attention_heads
        seq_len = size(x, 2)
        batch = size(x, 3)
        s = size(padding_mask_for_attn, 2)
        mask_col = permutedims(padding_mask_for_attn, (2, 1))
        pad_bias = ifelse.(mask_col, Float32(-Inf), 0f0)
        pad_bias_4d = reshape(pad_bias, s, 1, 1, batch)
        precomputed_attn_bias = repeat(pad_bias_4d, 1, seq_len, model.attention_heads, 1)
    end

    for layer_idx in 1:model.num_layers
        x, attn_weights = model.layers[layer_idx](
            x;
            self_attn_padding_mask = padding_mask_for_attn,
            need_head_weights = need_head_weights,
            _precomputed_attn_bias = precomputed_attn_bias,
        )
        if layer_idx in repr_set
            hidden_representations[layer_idx] = permutedims(x, (3, 2, 1))
        end
        if need_head_weights
            attn_weights_all[layer_idx] = attn_weights
        end
    end

    x = model.emb_layer_norm_after(x)
    if model.num_layers in repr_set
        hidden_representations[model.num_layers] = permutedims(x, (3, 2, 1))
    end

    logits = permutedims(model.lm_head(x), (3, 2, 1))

    attentions = nothing
    if need_head_weights
        # Stack to (B, L, H, T, T)
        expanded = map(attn -> reshape(attn, size(attn, 1), 1, size(attn, 2), size(attn, 3), size(attn, 4)), attn_weights_all)
        attentions = cat(expanded...; dims=2)
    end

    return ESM2Output(logits, hidden_representations, attentions)
end
