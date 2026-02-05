using NNlib

@concrete struct OFMultiheadAttention <: Onion.Layer
    linear_q
    linear_k
    linear_v
    linear_o
    linear_g
    no_heads::Int
    c_hidden::Int
    gating::Bool
    inf::Float32
end

@layer OFMultiheadAttention

function OFMultiheadAttention(c_q::Int, c_k::Int, c_v::Int, c_hidden::Int, no_heads::Int; gating::Bool=true, inf::Real=1e9)
    linear_q = LinearLast(c_q, c_hidden * no_heads; bias=false)
    linear_k = LinearLast(c_k, c_hidden * no_heads; bias=false)
    linear_v = LinearLast(c_v, c_hidden * no_heads; bias=false)
    linear_o = LinearLast(c_hidden * no_heads, c_q; bias=true)
    linear_g = gating ? LinearLast(c_q, c_hidden * no_heads; bias=true) : nothing
    return OFMultiheadAttention(linear_q, linear_k, linear_v, linear_o, linear_g, no_heads, c_hidden, gating, Float32(inf))
end

function (m::OFMultiheadAttention)(q_x::AbstractArray, kv_x::AbstractArray; biases::AbstractVector=Any[])
    # q_x: (..., Q, Cq), kv_x: (..., K, Ck)
    q = m.linear_q(q_x)
    k = m.linear_k(kv_x)
    v = m.linear_v(kv_x)

    # reshape to (..., Q, C_hidden, H) to match PyTorch view under column-major
    q = reshape(q, size(q)[1:end-1]..., m.c_hidden, m.no_heads)
    k = reshape(k, size(k)[1:end-1]..., m.c_hidden, m.no_heads)
    v = reshape(v, size(v)[1:end-1]..., m.c_hidden, m.no_heads)

    # permute to (..., Q, H, C_hidden)
    n = ndims(q)
    q = permutedims(q, vcat(collect(1:(n - 2)), n, n - 1))
    k = permutedims(k, vcat(collect(1:(n - 2)), n, n - 1))
    v = permutedims(v, vcat(collect(1:(n - 2)), n, n - 1))

    # transpose to (..., H, Q, C_hidden)
    n = ndims(q)
    q = permutedims(q, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))
    k = permutedims(k, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))
    v = permutedims(v, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))

    # scale
    q = q .* (1f0 / sqrt(Float32(m.c_hidden)))

    batch_shape = size(q_x)[1:end-2]
    Q = size(q_x, ndims(q_x) - 1)
    K = size(kv_x, ndims(kv_x) - 1)
    H = m.no_heads
    C = m.c_hidden

    q3 = reshape(q, prod(batch_shape) * H, Q, C)
    k3 = reshape(k, prod(batch_shape) * H, K, C)
    v3 = reshape(v, prod(batch_shape) * H, K, C)

    q3 = permutedims(q3, (2, 3, 1)) # (Q, C, B*H)
    k3 = permutedims(k3, (2, 3, 1)) # (K, C, B*H)
    v3 = permutedims(v3, (2, 3, 1)) # (K, C, B*H)

    a = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3))) # (Q, K, B*)

    if !isempty(biases)
        # reshape logits to (batch..., H, Q, K) for broadcasted bias addition
        a = reshape(a, Q, K, batch_shape..., m.no_heads)
        a = permutedims(a, (3:(2 + length(batch_shape))..., ndims(a), 1, 2)) # (batch..., H, Q, K)
        for bias in biases
            a = a .+ bias
        end
        # back to (Q, K, batch..., H)
        n = ndims(a)
        b = n - 3
        a = permutedims(a, vcat(b + 2, b + 3, collect(1:b), b + 1))
        a = reshape(a, Q, K, :)
    end

    a = NNlib.softmax(a; dims=2)

    o = NNlib.batched_mul(a, v3) # (Q, C, B*H)

    # reshape back to (..., H, Q, C_hidden)
    o = reshape(o, Q, C, batch_shape..., m.no_heads)
    o = permutedims(o, (3:(2 + length(batch_shape))..., ndims(o), 1, 2)) # (batch..., H, Q, C)
    # to (..., Q, H, C_hidden)
    o = permutedims(o, vcat(collect(1:(ndims(o) - 3)), [ndims(o) - 1, ndims(o) - 2, ndims(o)]))

    if m.gating
        g = NNlib.sigmoid.(m.linear_g(q_x))
        g = reshape(g, size(g)[1:end-1]..., m.c_hidden, m.no_heads)
        n = ndims(g)
        g = permutedims(g, vcat(collect(1:(n - 2)), n, n - 1)) # (..., Q, H, C_hidden)
        o .*= g
    end

    # flatten head and channel dims
    n = ndims(o)
    o = permutedims(o, vcat(collect(1:(n - 2)), n, n - 1)) # (..., Q, C_hidden, H)
    o = reshape(o, size(o)[1:end-2]..., m.no_heads * m.c_hidden)
    return m.linear_o(o)
end

@concrete struct OFMultiheadAttentionJL <: Onion.Layer
    linear_q
    linear_k
    linear_v
    linear_o
    linear_g
    no_heads::Int
    c_hidden::Int
    gating::Bool
    inf::Float32
end

@layer OFMultiheadAttentionJL

function OFMultiheadAttentionJL(c_q::Int, c_k::Int, c_v::Int, c_hidden::Int, no_heads::Int; gating::Bool=true, inf::Real=1e9)
    linear_q = LinearFirst(c_q, c_hidden * no_heads; bias=false)
    linear_k = LinearFirst(c_k, c_hidden * no_heads; bias=false)
    linear_v = LinearFirst(c_v, c_hidden * no_heads; bias=false)
    linear_o = LinearFirst(c_hidden * no_heads, c_q; bias=true)
    linear_g = gating ? LinearFirst(c_q, c_hidden * no_heads; bias=true) : nothing
    return OFMultiheadAttentionJL(linear_q, linear_k, linear_v, linear_o, linear_g, no_heads, c_hidden, gating, Float32(inf))
end

function (m::OFMultiheadAttentionJL)(q_x::AbstractArray, kv_x::AbstractArray; biases::AbstractVector=Any[])
    # q_x: (Cq, Q, batch...), kv_x: (Ck, K, batch...)
    q = m.linear_q(q_x)
    k = m.linear_k(kv_x)
    v = m.linear_v(kv_x)

    # reshape to (C_hidden, H, Q, batch...)
    q = reshape(q, m.c_hidden, m.no_heads, size(q, 2), size(q)[3:end]...)
    k = reshape(k, m.c_hidden, m.no_heads, size(k, 2), size(k)[3:end]...)
    v = reshape(v, m.c_hidden, m.no_heads, size(v, 2), size(v)[3:end]...)

    # permute to (batch..., H, Q, C_hidden)
    n = ndims(q)
    perm = vcat(4:n, 2, 3, 1)
    q = permutedims(q, perm)
    k = permutedims(k, perm)
    v = permutedims(v, perm)

    # scale
    q = q .* (1f0 / sqrt(Float32(m.c_hidden)))

    batch_shape = size(q_x)[3:end]
    B = prod(batch_shape)
    Q = size(q_x, 2)
    K = size(kv_x, 2)
    H = m.no_heads
    C = m.c_hidden

    q_flat = reshape(q, B, H, Q, C)
    k_flat = reshape(k, B, H, K, C)
    v_flat = reshape(v, B, H, K, C)

    q3 = reshape(permutedims(q_flat, (3, 4, 1, 2)), Q, C, B * H)
    k3 = reshape(permutedims(k_flat, (3, 4, 1, 2)), K, C, B * H)
    v3 = reshape(permutedims(v_flat, (3, 4, 1, 2)), K, C, B * H)

    a = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3))) # (Q, K, B*H)

    if !isempty(biases)
        a = reshape(a, Q, K, batch_shape..., H)
        a = permutedims(a, (3:(2 + length(batch_shape))..., ndims(a), 1, 2)) # (batch..., H, Q, K)
        for bias in biases
            a = a .+ bias
        end
        n = ndims(a)
        b = n - 3
        a = permutedims(a, vcat(b + 2, b + 3, collect(1:b), b + 1))
        a = reshape(a, Q, K, :)
    end

    a = NNlib.softmax(a; dims=2)

    o = NNlib.batched_mul(a, v3) # (Q, C, B*H)

    o = reshape(o, Q, C, batch_shape..., H)
    o = permutedims(o, (3:(2 + length(batch_shape))..., ndims(o), 1, 2)) # (batch..., H, Q, C)

    b = length(batch_shape)
    perm_out = vcat(b + 3, b + 1, b + 2, collect(1:b)) # (C, H, Q, batch...)
    o = permutedims(o, perm_out)
    o = reshape(o, m.c_hidden * m.no_heads, Q, batch_shape...)

    if m.gating
        g = NNlib.sigmoid.(m.linear_g(q_x))
        o = g .* o
    end

    return m.linear_o(o)
end

@concrete struct TriangleAttention <: Onion.Layer
    layer_norm
    linear
    mha
    starting::Bool
    inf::Float32
end

@layer TriangleAttention

function TriangleAttention(c_in::Int, c_hidden::Int, no_heads::Int; starting::Bool=true, inf::Real=1e9)
    layer_norm = LayerNormLast(c_in)
    linear = LinearLast(c_in, no_heads; bias=false)
    mha = OFMultiheadAttention(c_in, c_in, c_in, c_hidden, no_heads; gating=true, inf=inf)
    return TriangleAttention(layer_norm, linear, mha, starting, Float32(inf))
end

function (m::TriangleAttention)(x::AbstractArray; mask=nothing, chunk_size=nothing)
    if mask === nothing
        mask = ones_like(x, size(x)[1:end-1]...)
    end

    if !m.starting
        n = ndims(x)
        x = permutedims(x, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))
        nmask = ndims(mask)
        mask = permutedims(mask, vcat(collect(1:(nmask - 2)), [nmask, nmask - 1]))
    end

    x = m.layer_norm(x)

    # match OpenFold: mask bias is [*, I, 1, 1, J]
    mask_bias = m.inf .* (mask .- 1)
    mask_bias = reshape(mask_bias, size(mask_bias, 1), size(mask_bias, 2), 1, 1, size(mask_bias, 3))

    # match OpenFold: triangle bias is [*, 1, H, I, J]
    triangle_bias = permute_final_dims(m.linear(x), (2, 0, 1)) # (B, H, I, J)
    triangle_bias = reshape(
        triangle_bias,
        size(triangle_bias, 1),
        1,
        size(triangle_bias, 2),
        size(triangle_bias, 3),
        size(triangle_bias, 4),
    )

    biases = [mask_bias, triangle_bias]
    out = m.mha(x, x; biases=biases)

    if !m.starting
        n = ndims(out)
        out = permutedims(out, vcat(collect(1:(n - 3)), [n - 1, n - 2, n]))
    end

    return out
end

@concrete struct TriangleAttentionJL <: Onion.Layer
    layer_norm
    linear
    mha
    starting::Bool
    inf::Float32
end

@layer TriangleAttentionJL

function TriangleAttentionJL(c_in::Int, c_hidden::Int, no_heads::Int; starting::Bool=true, inf::Real=1e9)
    layer_norm = LayerNormFirst(c_in)
    linear = LinearFirst(c_in, no_heads; bias=false)
    mha = OFMultiheadAttentionJL(c_in, c_in, c_in, c_hidden, no_heads; gating=true, inf=inf)
    return TriangleAttentionJL(layer_norm, linear, mha, starting, Float32(inf))
end

function (m::TriangleAttentionJL)(x::AbstractArray; mask=nothing, chunk_size=nothing)
    # x: (C, I, J, B)
    if mask === nothing
        mask = ones_like(x, size(x, 2), size(x, 3), size(x, 4))
    end

    if m.starting
        # batch dims: (B, I), Q = J
        x_att = permutedims(x, (1, 3, 4, 2)) # (C, J, B, I)
        mask_att = permutedims(mask, (2, 3, 1)) # (J, B, I)
    else
        # batch dims: (B, J), Q = I
        x_att = permutedims(x, (1, 2, 4, 3)) # (C, I, B, J)
        mask_att = permutedims(mask, (1, 3, 2)) # (I, B, J)
    end

    x_att = m.layer_norm(x_att)

    # mask bias: (batch..., 1, 1, K)
    mask_bias = m.inf .* (mask_att .- 1)
    mask_bias = permutedims(mask_bias, (2, 3, 1)) # (batch_dim1, batch_dim2, K)
    mask_bias = reshape(mask_bias, size(mask_bias, 1), size(mask_bias, 2), 1, 1, size(mask_bias, 3))

    # triangle bias: (B, 1, H, batch_dim2, Q)
    triangle_bias = m.linear(x_att) # (H, Q, B, batch_dim2)
    triangle_bias = permutedims(triangle_bias, (3, 1, 4, 2)) # (B, H, batch_dim2, Q)
    triangle_bias = reshape(triangle_bias, size(triangle_bias, 1), 1, size(triangle_bias, 2), size(triangle_bias, 3), size(triangle_bias, 4))

    biases = [mask_bias, triangle_bias]
    out = m.mha(x_att, x_att; biases=biases)

    if m.starting
        out = permutedims(out, (1, 4, 2, 3)) # (C, I, J, B)
    else
        out = permutedims(out, (1, 2, 4, 3)) # (C, I, J, B)
    end

    return out
end

@concrete struct TriangleMultiplicativeUpdate <: Onion.Layer
    linear_a_p
    linear_a_g
    linear_b_p
    linear_b_g
    linear_g
    linear_z
    layer_norm_in
    layer_norm_out
    outgoing::Bool
end

@layer TriangleMultiplicativeUpdate

function TriangleMultiplicativeUpdate(c_z::Int, c_hidden::Int; outgoing::Bool=true)
    linear_a_p = LinearLast(c_z, c_hidden)
    linear_a_g = LinearLast(c_z, c_hidden)
    linear_b_p = LinearLast(c_z, c_hidden)
    linear_b_g = LinearLast(c_z, c_hidden)
    linear_g = LinearLast(c_z, c_z)
    linear_z = LinearLast(c_hidden, c_z)
    layer_norm_in = LayerNormLast(c_z)
    layer_norm_out = LayerNormLast(c_hidden)
    return TriangleMultiplicativeUpdate(
        linear_a_p,
        linear_a_g,
        linear_b_p,
        linear_b_g,
        linear_g,
        linear_z,
        layer_norm_in,
        layer_norm_out,
        outgoing,
    )
end

function _combine_projections(a::AbstractArray, b::AbstractArray, outgoing::Bool)
    if outgoing
        a = permute_final_dims(a, (2, 0, 1))
        b = permute_final_dims(b, (2, 1, 0))
    else
        a = permute_final_dims(a, (2, 1, 0))
        b = permute_final_dims(b, (2, 0, 1))
    end

    # a, b: (B, C, L, L)
    B = size(a, 1) * size(a, 2)
    L = size(a, 3)
    a2 = reshape(a, B, L, L)
    b2 = reshape(b, B, L, L)
    a3 = permutedims(a2, (2, 3, 1))
    b3 = permutedims(b2, (2, 3, 1))
    x3 = NNlib.batched_mul(a3, b3)
    x2 = permutedims(x3, (3, 1, 2))
    x = reshape(x2, size(a, 1), size(a, 2), L, L)

    return permute_final_dims(x, (1, 2, 0))
end

function (m::TriangleMultiplicativeUpdate)(z::AbstractArray; mask=nothing)
    if mask === nothing
        mask = ones_like(z, size(z)[1:end-1]...)
    end
    mask = reshape(mask, size(mask)..., 1)
    z_norm = m.layer_norm_in(z)
    a = mask .* NNlib.sigmoid.(m.linear_a_g(z_norm)) .* m.linear_a_p(z_norm)
    b = mask .* NNlib.sigmoid.(m.linear_b_g(z_norm)) .* m.linear_b_p(z_norm)
    x = _combine_projections(a, b, m.outgoing)
    x = m.layer_norm_out(x)
    x = m.linear_z(x)
    g = NNlib.sigmoid.(m.linear_g(z_norm))
    return x .* g
end

@concrete struct TriangleMultiplicativeUpdateJL <: Onion.Layer
    linear_a_p
    linear_a_g
    linear_b_p
    linear_b_g
    linear_g
    linear_z
    layer_norm_in
    layer_norm_out
    outgoing::Bool
end

@layer TriangleMultiplicativeUpdateJL

function TriangleMultiplicativeUpdateJL(c_z::Int, c_hidden::Int; outgoing::Bool=true)
    linear_a_p = LinearFirst(c_z, c_hidden)
    linear_a_g = LinearFirst(c_z, c_hidden)
    linear_b_p = LinearFirst(c_z, c_hidden)
    linear_b_g = LinearFirst(c_z, c_hidden)
    linear_g = LinearFirst(c_z, c_z)
    linear_z = LinearFirst(c_hidden, c_z)
    layer_norm_in = LayerNormFirst(c_z)
    layer_norm_out = LayerNormFirst(c_hidden)
    return TriangleMultiplicativeUpdateJL(
        linear_a_p,
        linear_a_g,
        linear_b_p,
        linear_b_g,
        linear_g,
        linear_z,
        layer_norm_in,
        layer_norm_out,
        outgoing,
    )
end

function _combine_projections_jl(a::AbstractArray, b::AbstractArray, outgoing::Bool)
    # a, b: (C, L, L, B)
    a_perm = permutedims(a, (2, 3, 1, 4)) # (L, L, C, B)
    b_perm = permutedims(b, (2, 3, 1, 4)) # (L, L, C, B)
    L = size(a_perm, 1)
    C = size(a_perm, 3)
    B = size(a_perm, 4)
    a3 = reshape(a_perm, L, L, C * B)
    b3 = reshape(b_perm, L, L, C * B)
    if outgoing
        x3 = NNlib.batched_mul(a3, permutedims(b3, (2, 1, 3)))
    else
        x3 = NNlib.batched_mul(permutedims(a3, (2, 1, 3)), b3)
    end
    x2 = reshape(x3, L, L, C, B)
    x = permutedims(x2, (3, 1, 2, 4)) # (C, L, L, B)
    return x
end

function (m::TriangleMultiplicativeUpdateJL)(z::AbstractArray; mask=nothing)
    # z: (C_z, L, L, B)
    if mask === nothing
        mask = ones_like(z, size(z, 2), size(z, 3), size(z, 4))
    end
    mask = reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
    z_norm = m.layer_norm_in(z)
    a = mask .* NNlib.sigmoid.(m.linear_a_g(z_norm)) .* m.linear_a_p(z_norm)
    b = mask .* NNlib.sigmoid.(m.linear_b_g(z_norm)) .* m.linear_b_p(z_norm)
    x = _combine_projections_jl(a, b, m.outgoing)
    x = m.layer_norm_out(x)
    x = m.linear_z(x)
    g = NNlib.sigmoid.(m.linear_g(z_norm))
    return x .* g
end

struct TriangleMultiplicationOutgoing <: Onion.Layer
    inner::TriangleMultiplicativeUpdate
end

@layer TriangleMultiplicationOutgoing

function TriangleMultiplicationOutgoing(c_z::Int, c_hidden::Int)
    return TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate(c_z, c_hidden; outgoing=true))
end

(m::TriangleMultiplicationOutgoing)(z; mask=nothing) = m.inner(z; mask=mask)

struct TriangleMultiplicationIncoming <: Onion.Layer
    inner::TriangleMultiplicativeUpdate
end

@layer TriangleMultiplicationIncoming

function TriangleMultiplicationIncoming(c_z::Int, c_hidden::Int)
    return TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate(c_z, c_hidden; outgoing=false))
end

(m::TriangleMultiplicationIncoming)(z; mask=nothing) = m.inner(z; mask=mask)

struct TriangleMultiplicationOutgoingJL <: Onion.Layer
    inner::TriangleMultiplicativeUpdateJL
end

@layer TriangleMultiplicationOutgoingJL

function TriangleMultiplicationOutgoingJL(c_z::Int, c_hidden::Int)
    return TriangleMultiplicationOutgoingJL(TriangleMultiplicativeUpdateJL(c_z, c_hidden; outgoing=true))
end

(m::TriangleMultiplicationOutgoingJL)(z; mask=nothing) = m.inner(z; mask=mask)

struct TriangleMultiplicationIncomingJL <: Onion.Layer
    inner::TriangleMultiplicativeUpdateJL
end

@layer TriangleMultiplicationIncomingJL

function TriangleMultiplicationIncomingJL(c_z::Int, c_hidden::Int)
    return TriangleMultiplicationIncomingJL(TriangleMultiplicativeUpdateJL(c_z, c_hidden; outgoing=false))
end

(m::TriangleMultiplicationIncomingJL)(z; mask=nothing) = m.inner(z; mask=mask)

@concrete struct TriangularSelfAttentionBlock <: Onion.Layer
    layernorm_1
    sequence_to_pair
    pair_to_sequence
    seq_attention
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    mlp_seq
    mlp_pair
    drop
    row_drop
    col_drop
end

@layer TriangularSelfAttentionBlock

function TriangularSelfAttentionBlock(
    sequence_state_dim::Int,
    pairwise_state_dim::Int,
    sequence_head_width::Int,
    pairwise_head_width::Int;
    dropout::Real=0,
)
    sequence_num_heads = sequence_state_dim ÷ sequence_head_width
    pairwise_num_heads = pairwise_state_dim ÷ pairwise_head_width

    layernorm_1 = LayerNormLast(sequence_state_dim)
    sequence_to_pair = SequenceToPair(sequence_state_dim, pairwise_state_dim ÷ 2, pairwise_state_dim)
    pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads)

    seq_attention = ESMFoldAttention(sequence_state_dim, sequence_num_heads, sequence_head_width; gated=true)

    tri_mul_out = TriangleMultiplicationOutgoing(pairwise_state_dim, pairwise_state_dim)
    tri_mul_in = TriangleMultiplicationIncoming(pairwise_state_dim, pairwise_state_dim)
    tri_att_start = TriangleAttention(pairwise_state_dim, pairwise_head_width, pairwise_num_heads; starting=true, inf=1e9)
    tri_att_end = TriangleAttention(pairwise_state_dim, pairwise_head_width, pairwise_num_heads; starting=false, inf=1e9)

    mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim; dropout=dropout)
    mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim; dropout=dropout)

    drop = SharedDropout(dropout, 1)
    row_drop = SharedDropout(dropout * 2, 2)
    col_drop = SharedDropout(dropout * 2, 1)

    return TriangularSelfAttentionBlock(
        layernorm_1,
        sequence_to_pair,
        pair_to_sequence,
        seq_attention,
        tri_mul_out,
        tri_mul_in,
        tri_att_start,
        tri_att_end,
        mlp_seq,
        mlp_pair,
        drop,
        row_drop,
        col_drop,
    )
end

function (m::TriangularSelfAttentionBlock)(sequence_state, pairwise_state; mask=nothing, chunk_size=nothing, residue_index=nothing)
    bias = m.pair_to_sequence(pairwise_state)
    y = m.layernorm_1(sequence_state)
    y, _ = m.seq_attention(y; mask=mask, bias=bias)
    sequence_state = sequence_state .+ m.drop(y)
    sequence_state = m.mlp_seq(sequence_state)

    pairwise_state = pairwise_state .+ m.sequence_to_pair(sequence_state)

    tri_mask = mask === nothing ? nothing :
        (reshape(mask, size(mask, 1), size(mask, 2), 1) .* reshape(mask, size(mask, 1), 1, size(mask, 2)))
    pairwise_state = pairwise_state .+ m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))

    pairwise_state = m.mlp_pair(pairwise_state)

    return sequence_state, pairwise_state
end

@concrete struct TriangularSelfAttentionBlockJL <: Onion.Layer
    layernorm_1
    sequence_to_pair
    pair_to_sequence
    seq_attention
    tri_mul_out
    tri_mul_in
    tri_att_start
    tri_att_end
    mlp_seq
    mlp_pair
    drop
    row_drop
    col_drop
end

@layer TriangularSelfAttentionBlockJL

function TriangularSelfAttentionBlockJL(
    sequence_state_dim::Int,
    pairwise_state_dim::Int,
    sequence_head_width::Int,
    pairwise_head_width::Int;
    dropout::Real=0,
)
    sequence_num_heads = sequence_state_dim ÷ sequence_head_width
    pairwise_num_heads = pairwise_state_dim ÷ pairwise_head_width

    layernorm_1 = LayerNormFirst(sequence_state_dim)
    sequence_to_pair = SequenceToPairJL(sequence_state_dim, pairwise_state_dim ÷ 2, pairwise_state_dim)
    pair_to_sequence = PairToSequenceJL(pairwise_state_dim, sequence_num_heads)

    seq_attention = ESMFoldAttentionJL(sequence_state_dim, sequence_num_heads, sequence_head_width; gated=true)

    tri_mul_out = TriangleMultiplicationOutgoingJL(pairwise_state_dim, pairwise_state_dim)
    tri_mul_in = TriangleMultiplicationIncomingJL(pairwise_state_dim, pairwise_state_dim)
    tri_att_start = TriangleAttentionJL(pairwise_state_dim, pairwise_head_width, pairwise_num_heads; starting=true, inf=1e9)
    tri_att_end = TriangleAttentionJL(pairwise_state_dim, pairwise_head_width, pairwise_num_heads; starting=false, inf=1e9)

    mlp_seq = ResidueMLPJL(sequence_state_dim, 4 * sequence_state_dim; dropout=dropout)
    mlp_pair = ResidueMLPJL(pairwise_state_dim, 4 * pairwise_state_dim; dropout=dropout)

    drop = SharedDropout(dropout, 3)
    row_drop = SharedDropout(dropout * 2, 2)
    col_drop = SharedDropout(dropout * 2, 1)

    return TriangularSelfAttentionBlockJL(
        layernorm_1,
        sequence_to_pair,
        pair_to_sequence,
        seq_attention,
        tri_mul_out,
        tri_mul_in,
        tri_att_start,
        tri_att_end,
        mlp_seq,
        mlp_pair,
        drop,
        row_drop,
        col_drop,
    )
end

function (m::TriangularSelfAttentionBlockJL)(sequence_state, pairwise_state; mask=nothing, chunk_size=nothing, residue_index=nothing)
    # sequence_state: (C_s, L, B), pairwise_state: (C_z, L, L, B)
    bias = m.pair_to_sequence(pairwise_state) # (H, L, L, B)
    y = m.layernorm_1(sequence_state)
    y, _ = m.seq_attention(y; mask=mask, bias=bias)
    sequence_state = sequence_state .+ m.drop(y)
    sequence_state = m.mlp_seq(sequence_state)

    pairwise_state = pairwise_state .+ m.sequence_to_pair(sequence_state)

    tri_mask = mask === nothing ? nothing :
        (reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2)))
    pairwise_state = pairwise_state .+ m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
    pairwise_state = pairwise_state .+ m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))

    pairwise_state = m.mlp_pair(pairwise_state)

    return sequence_state, pairwise_state
end
