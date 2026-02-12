"""
    esm2_forward_ad(esm::ESM2, tokens_bt::AbstractArray{Int,2}) → AbstractArray

AD-compatible forward pass through ESM2, returning final hidden states.

The standard ESM2 forward uses in-place ops (`.+=`, `.=`) that Zygote cannot differentiate.
This function replaces all in-place operations with allocating equivalents, enabling
`Zygote.gradient` through the full 33-layer transformer.

# Arguments
- `esm::ESM2`: The ESM2 language model.
- `tokens_bt::AbstractArray{Int,2}`: Token indices in `(B, T)` layout, 0-indexed
  (matching `Alphabet` conventions where `padding_idx` is typically 1).

# Returns
- `AbstractArray` of shape `(C, T, B)` — final hidden states after `emb_layer_norm_after`.

# Example
```julia
using Zygote
grads = Zygote.gradient(model.embed.esm) do esm
    x = esm2_forward_ad(esm, tokens_bt)
    sum(x)
end
```

# Notes
- Token dropout is NOT applied (inference path only).
- Per-layer representations are NOT collected; only the final output is returned.
- Padding mask and precomputed attention bias are wrapped in `@ignore_derivatives`
  so they are treated as opaque constants by the AD system.
"""
function esm2_forward_ad(esm::ESM2, tokens_bt::AbstractArray{Int,2})
    pad_idx = esm.alphabet.padding_idx

    tokens_tb = permutedims(tokens_bt, (2, 1))  # (T, B)
    x = esm.embed_scale .* esm.embed_tokens(tokens_tb .+ 1)

    # Padding mask — treat as constant for AD
    padding_mask_bt = @ignore_derivatives tokens_bt .== pad_idx
    has_padding = @ignore_derivatives any(padding_mask_bt)

    # Apply padding mask (allocating, not in-place)
    if has_padding
        padding_mask_tb = @ignore_derivatives permutedims(padding_mask_bt, (2, 1))
        x = x .* reshape(1 .- padding_mask_tb, 1, size(x, 2), size(x, 3))
    end

    # Pre-compute attention bias (constant — no gradient needed)
    precomputed_attn_bias = @ignore_derivatives begin
        if has_padding
            pad_tb = permutedims(padding_mask_bt, (2, 1))
            pad_bias = ifelse.(pad_tb, Float32(-Inf), 0f0)
            pad_bias_4d = reshape(pad_bias, size(pad_bias, 1), 1, 1, size(pad_bias, 2))
            repeat(pad_bias_4d, 1, size(tokens_tb, 1), esm.attention_heads, 1)
        else
            nothing
        end
    end

    padding_mask_for_attn = @ignore_derivatives has_padding ? padding_mask_bt : nothing

    # Run all transformer layers (allocating residual path — no .+=)
    for layer_idx in 1:esm.num_layers
        layer = esm.layers[layer_idx]

        residual = x
        x = layer.self_attn_layer_norm(x)
        attn_out, _ = layer.self_attn(
            x;
            key_padding_mask = padding_mask_for_attn,
            need_head_weights = false,
            _precomputed_attn_bias = precomputed_attn_bias,
        )
        x = residual .+ attn_out

        residual = x
        x = layer.final_layer_norm(x)
        x = NNlib.gelu.(layer.fc1(x))
        x = layer.fc2(x)
        x = residual .+ x
    end

    x = esm.emb_layer_norm_after(x)
    return x  # (C, T, B)
end
