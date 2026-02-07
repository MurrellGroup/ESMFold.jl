# ESMFoldEmbedConfig and LayerNormMLP are imported from Onion

@concrete struct ESMFoldEmbed <: Onion.Layer
    esm::ESM2
    esm_dict::Alphabet
    esm_feats::Int
    esm_attns::Int
    af2_to_esm::AbstractVector{Int}
    esm_s_combine::AbstractVector{Float32}
    esm_s_mlp::LayerNormMLP
    esm_z_mlp::Union{LayerNormMLP,Nothing}
    n_tokens_embed::Int
    pad_idx::Int
    unk_idx::Int
    mask_idx::Int
    embedding
    cfg::ESMFoldEmbedConfig
end

@layer ESMFoldEmbed

function _af2_to_esm(alphabet::Alphabet)
    esm_reorder = [alphabet.padding_idx]
    for restype in RESTYPE_TABLE.restypes_with_x
        push!(esm_reorder, get_idx(alphabet, restype))
    end
    return esm_reorder
end

function ESMFoldEmbed(
    esm::ESM2;
    c_s::Int,
    c_z::Int,
    use_esm_attn_map::Bool = false,
)
    esm_feats = esm.embed_dim
    esm_attns = esm.num_layers * esm.attention_heads
    esm_dict = esm.alphabet
    af2_to_esm = _af2_to_esm(esm_dict)
    esm_s_combine = zeros(Float32, esm.num_layers + 1)
    esm_s_mlp = LayerNormMLP(esm_feats, c_s)
    esm_z_mlp = use_esm_attn_map ? LayerNormMLP(esm_attns, c_z) : nothing

    n_tokens_embed = length(RESTYPE_TABLE.restypes) + 3
    pad_idx = 0
    unk_idx = n_tokens_embed - 2
    mask_idx = n_tokens_embed - 1
    embedding = Flux.Embedding(n_tokens_embed, c_s)

    cfg = ESMFoldEmbedConfig(c_s, c_z, use_esm_attn_map)

    return ESMFoldEmbed(
        esm,
        esm_dict,
        esm_feats,
        esm_attns,
        af2_to_esm,
        esm_s_combine,
        esm_s_mlp,
        esm_z_mlp,
        n_tokens_embed,
        pad_idx,
        unk_idx,
        mask_idx,
        embedding,
        cfg,
    )
end

function _af2_idx_to_esm_idx(m::ESMFoldEmbed, aa::AbstractArray{Int,2}, mask)
    af2_to_esm = typeof(aa) == typeof(m.af2_to_esm) ? m.af2_to_esm : to_device(m.af2_to_esm, aa, Int)
    aa_shift = aa .+ 1
    aa_masked = if mask === nothing
        aa_shift
    else
        if eltype(mask) <: Bool
            ifelse.(mask, aa_shift, 0)
        else
            ifelse.(mask .== 1, aa_shift, 0)
        end
    end
    return af2_to_esm[aa_masked .+ 1]
end

function _mask_inputs_to_esm(m::ESMFoldEmbed, esmaa, pattern)
    if pattern === nothing
        return esmaa
    end
    return ifelse.(pattern .== 1, m.esm_dict.mask_idx, esmaa)
end

function _pad_batch(seqs::AbstractVector{<:AbstractVector{<:Integer}}; pad_value::Int)
    lengths = map(length, seqs)
    isempty(lengths) && error("Expected at least one sequence.")
    maxlen = maximum(lengths)
    maxlen == 0 && error("Expected at least one non-empty sequence.")

    batch = length(seqs)
    aa = fill(pad_value, batch, maxlen)
    mask = zeros(Int, batch, maxlen)

    for (i, seq) in enumerate(seqs)
        n = length(seq)
        n == 0 && continue
        aa[i, 1:n] .= seq
        mask[i, 1:n] .= 1
    end

    return aa, mask
end

function _compute_language_model_representations(
    m::ESMFoldEmbed,
    esmaa::AbstractArray{Int,2};
    need_attn::Bool,
)
    batch_size = size(esmaa, 1)
    bosi = m.esm_dict.cls_idx
    eosi = m.esm_dict.eos_idx
    pad = m.esm_dict.padding_idx

    bos = to_device(fill(bosi, batch_size, 1), esmaa, eltype(esmaa))
    eos = to_device(fill(pad, batch_size, 1), esmaa, eltype(esmaa))
    esmaa2 = hcat(bos, esmaa, eos)

    lengths = sum(esmaa2 .!= pad, dims=2)
    positions = to_device(reshape(1:size(esmaa2, 2), 1, :), esmaa2, Int)
    eos_mask = positions .== (lengths .+ 1)
    esmaa2 = ifelse.(eos_mask, eosi, esmaa2)

    res = m.esm(
        esmaa2;
        repr_layers = collect(0:m.esm.num_layers),
        need_head_weights = need_attn,
    )

    num_layers = m.esm.num_layers + 1
    first_rep = res.representations[0]
    reps = zeros_like(first_rep, Float32, batch_size, size(esmaa2, 2), num_layers, m.esm.embed_dim)
    for layer in 0:m.esm.num_layers
        reps[:, :, layer + 1, :] .= Float32.(res.representations[layer])
    end
    # remove bos/eos
    reps = reps[:, 2:(end - 1), :, :]

    attn = nothing
    if need_attn
        attn = res.attentions
    end

    return reps, attn
end

function (m::ESMFoldEmbed)(
    aa::AbstractArray{Int,2};
    mask = nothing,
    masking_pattern = nothing,
    return_pair::Bool = false,
)
    if mask === nothing
        mask = ones(Int, size(aa))
    end

    esmaa = _af2_idx_to_esm_idx(m, aa, mask)
    esmaa = _mask_inputs_to_esm(m, esmaa, masking_pattern)

    esm_s, esm_attn = _compute_language_model_representations(
        m,
        esmaa;
        need_attn = return_pair && m.cfg.use_esm_attn_map,
    )

    weights = to_device(m.esm_s_combine, esm_s, eltype(esm_s))
    weights = NNlib.softmax(weights)
    weights_view = reshape(weights, 1, 1, length(weights), 1)
    esm_s_weighted = sum(esm_s .* weights_view, dims=3)
    esm_s_weighted = dropdims(esm_s_weighted, dims=3)

    esm_s_cf = permutedims(esm_s_weighted, (3, 2, 1))
    s_s_0 = m.esm_s_mlp(esm_s_cf)

    aa_tb = permutedims(aa, (2, 1))
    emb = m.embedding(aa_tb .+ 1)
    s_s_0 .= s_s_0 .+ emb

    if return_pair
        s_z_0 = nothing
        if m.cfg.use_esm_attn_map
            @assert esm_attn !== nothing
            attn = esm_attn
            attn = permutedims(attn, (1, 4, 5, 2, 3))
            b, t, _, l, h = size(attn)
            attn = reshape(attn, b, t, t, l * h)
            attn = attn[:, 2:(end - 1), 2:(end - 1), :]
            attn_cf = permutedims(attn, (4, 2, 3, 1))
            s_z_0 = m.esm_z_mlp(attn_cf)
        end
        return (
            sequence = s_s_0,
            pair = s_z_0,
        )
    end
    return s_s_0
end

function (m::ESMFoldEmbed)(
    seqs::AbstractVector{<:AbstractVector{<:Integer}};
    masking_pattern = nothing,
    return_pair::Bool = false,
)
    aa, mask = _pad_batch(seqs; pad_value = m.pad_idx)
    return m(
        aa;
        mask = mask,
        masking_pattern = masking_pattern,
        return_pair = return_pair,
    )
end

function (m::ESMFoldEmbed)(
    seqs::AbstractVector{<:AbstractString};
    masking_pattern = nothing,
    return_pair::Bool = false,
)
    aa_seqs = map(sequence_to_af2_indices, seqs)
    return m(aa_seqs; masking_pattern = masking_pattern, return_pair = return_pair)
end

function (m::ESMFoldEmbed)(
    seq::AbstractString;
    masking_pattern = nothing,
    return_pair::Bool = false,
)
    return m([seq]; masking_pattern = masking_pattern, return_pair = return_pair)
end
