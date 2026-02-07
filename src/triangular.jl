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
    linear_q = LinearFirst(c_q, c_hidden * no_heads; bias=false)
    linear_k = LinearFirst(c_k, c_hidden * no_heads; bias=false)
    linear_v = LinearFirst(c_v, c_hidden * no_heads; bias=false)
    linear_o = LinearFirst(c_hidden * no_heads, c_q; bias=true)
    linear_g = gating ? LinearFirst(c_q, c_hidden * no_heads; bias=true) : nothing
    return OFMultiheadAttention(linear_q, linear_k, linear_v, linear_o, linear_g, no_heads, c_hidden, gating, Float32(inf))
end

function (m::OFMultiheadAttention)(q_x::AbstractArray, kv_x::AbstractArray;
        biases::AbstractVector=Any[], _attn_bias_flash=nothing)
    # q_x: (Cq, Q, batch...), kv_x: (Ck, K, batch...)
    q = m.linear_q(q_x)
    k = m.linear_k(kv_x)
    v = m.linear_v(kv_x)

    C = m.c_hidden
    H = m.no_heads
    Q = size(q_x, 2)
    K = size(kv_x, 2)
    batch_shape = size(q_x)[3:end]
    B = prod(batch_shape)
    n_batch = length(batch_shape)

    # Fast CUDA path: go directly from linear output to flash attention format
    if isa(q, CuArray)
        # reshape to (C, H, Q/K, batch...) then permute to (C, Q/K, H, batch...) then flatten batch
        q_r = reshape(q, C, H, Q, batch_shape...)
        k_r = reshape(k, C, H, K, batch_shape...)
        v_r = reshape(v, C, H, K, batch_shape...)

        perm_in = Tuple(vcat(1, 3, 2, collect(4:(3 + n_batch))))  # (C, Q, H, batch...)
        # Use pre-allocated buffers with permutedims! to avoid allocation overhead
        q_dst_shape = (C, Q, H, batch_shape...)
        k_dst_shape = (C, K, H, batch_shape...)
        q_buf = _get_perm_buf(1, q_dst_shape)
        k_buf = _get_perm_buf(2, k_dst_shape)
        v_buf = _get_perm_buf(3, k_dst_shape)
        permutedims!(q_buf, q_r, perm_in)
        permutedims!(k_buf, k_r, perm_in)
        permutedims!(v_buf, v_r, perm_in)
        # Free linear projection outputs now that data is in pre-allocated buffers (~25.5MB)
        CUDA.unsafe_free!(q)
        CUDA.unsafe_free!(k)
        CUDA.unsafe_free!(v)
        q4 = reshape(q_buf, C, Q, H, B)
        k4 = reshape(k_buf, C, K, H, B)
        v4 = reshape(v_buf, C, K, H, B)

        # Scaling folded into flash attention scale parameter (saves 1 broadcast kernel)
        attn_scale = 1f0 / sqrt(Float32(C))
        # Pre-allocated flash attention output buffer
        flash_out = _get_perm_buf(5, (C, Q, H, B))

        if _attn_bias_flash !== nothing
            # Pre-computed bias in flash format (K, Q, H, bias_batch) — supports broadcast
            out4 = flash_attention_bias(q4, k4, v4, _attn_bias_flash; scale=attn_scale, output=flash_out)
        elseif !isempty(biases)
            # Handle biases: accumulate in (batch..., H, Q, K) then convert to (K, Q, H, B)
            # Accumulate biases at full shape to handle broadcasting
            bias_acc = zeros_like(q4, eltype(q4), batch_shape..., H, Q, K)
            for bias in biases
                bias_acc = bias_acc .+ bias
            end
            # (batch..., H, Q, K) → (K, Q, H, batch...) → reshape to (K, Q, H, B)
            perm_bias = vcat(n_batch + 3, n_batch + 2, n_batch + 1, collect(1:n_batch))
            bias4 = reshape(permutedims(bias_acc, perm_bias), K, Q, H, B)
            out4 = flash_attention_bias(q4, k4, v4, bias4; scale=attn_scale, output=flash_out)
        else
            out4 = flash_attention(q4, k4, v4; scale=attn_scale, output=flash_out)
        end

        # Output: (C, Q, H, B) → unflatten batch → (C, H, Q, batch...) → (C*H, Q, batch...)
        out_r = reshape(out4, C, Q, H, batch_shape...)
        perm_out = Tuple(vcat(1, 3, 2, collect(4:(3 + n_batch))))  # (C, H, Q, batch...)
        out_dst_shape = (C, H, Q, batch_shape...)
        out_buf = _get_perm_buf(4, out_dst_shape)
        permutedims!(out_buf, out_r, perm_out)
        o = reshape(out_buf, C * H, Q, batch_shape...)

        if m.gating
            # Fuse sigmoid + multiply into single broadcast kernel
            g_raw = m.linear_g(q_x)
            @. o = NNlib.sigmoid(g_raw) * o
            CUDA.unsafe_free!(g_raw)  # ~8.5MB freed immediately
        end

        return m.linear_o(o)
    end

    # CPU fallback: standard batched_mul path

    q = reshape(q, C, H, Q, batch_shape...)
    k = reshape(k, C, H, K, batch_shape...)
    v = reshape(v, C, H, K, batch_shape...)

    n = ndims(q)
    perm = vcat(4:n, 2, 3, 1)
    q = permutedims(q, perm)
    k = permutedims(k, perm)
    v = permutedims(v, perm)

    q = q .* (1f0 / sqrt(Float32(C)))

    q_flat = reshape(q, B, H, Q, C)
    k_flat = reshape(k, B, H, K, C)
    v_flat = reshape(v, B, H, K, C)

    q3 = reshape(permutedims(q_flat, (3, 4, 1, 2)), Q, C, B * H)
    k3 = reshape(permutedims(k_flat, (3, 4, 1, 2)), K, C, B * H)
    v3 = reshape(permutedims(v_flat, (3, 4, 1, 2)), K, C, B * H)

    a = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3)))

    if !isempty(biases)
        a = reshape(a, Q, K, batch_shape..., H)
        a = permutedims(a, (3:(2 + n_batch)..., ndims(a), 1, 2))
        for bias in biases
            a = a .+ bias
        end
        n = ndims(a)
        b = n - 3
        a = permutedims(a, vcat(b + 2, b + 3, collect(1:b), b + 1))
        a = reshape(a, Q, K, :)
    end

    a = NNlib.softmax(a; dims=2)
    o = NNlib.batched_mul(a, v3)

    o = reshape(o, Q, C, batch_shape..., H)
    o = permutedims(o, (3:(2 + n_batch)..., ndims(o), 1, 2))

    b = n_batch
    perm_out = vcat(b + 3, b + 1, b + 2, collect(1:b))
    o = permutedims(o, perm_out)
    o = reshape(o, C * H, Q, batch_shape...)

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
    layer_norm = LayerNormFirst(c_in)
    linear = LinearFirst(c_in, no_heads; bias=false)
    mha = OFMultiheadAttention(c_in, c_in, c_in, c_hidden, no_heads; gating=true, inf=inf)
    return TriangleAttention(layer_norm, linear, mha, starting, Float32(inf))
end

function (m::TriangleAttention)(x::AbstractArray; mask=nothing, chunk_size=nothing)
    # x: (C, I, J, B)
    if isa(x, CuArray)
        # Pre-allocated entry permutedims buffer (avoids ~8.5MB allocation per call)
        att_shape = m.starting ? (size(x,1), size(x,3), size(x,4), size(x,2)) :
                                 (size(x,1), size(x,2), size(x,4), size(x,3))
        x_att = _get_perm_buf(10, att_shape)
        if m.starting
            permutedims!(x_att, x, (1, 3, 4, 2)) # (C, J, B, I)
        else
            permutedims!(x_att, x, (1, 2, 4, 3)) # (C, I, B, J)
        end
        layernorm_inplace!(m.layer_norm, x_att)
    else
        if m.starting
            x_att = permutedims(x, (1, 3, 4, 2))
        else
            x_att = permutedims(x, (1, 2, 4, 3))
        end
        x_att = m.layer_norm(x_att)
    end

    # CUDA fast path: construct bias directly in flash attention format
    # The triangle bias is independent of the last batch dimension, so we can use
    # broadcast bias (K, Q, H, B_orig) instead of full (K, Q, H, B_orig*batch_dim2).
    # This saves ~33MB per call by avoiding the 5D intermediate allocation.
    if isa(x_att, CuArray) && mask === nothing
        # triangle_bias_raw: (H, seq_dim, B, batch_dim2)
        # bias4[k, q, h, b] = raw[h, k, b, q] → permutedims(raw, (2, 4, 1, 3))
        tb_raw = m.linear(x_att)
        bias_shape = (size(tb_raw,2), size(tb_raw,4), size(tb_raw,1), size(tb_raw,3))
        bias_flash = _get_perm_buf(11, bias_shape)
        permutedims!(bias_flash, tb_raw, (2, 4, 1, 3))  # (K=seq, Q=batch2, H, B)
        CUDA.unsafe_free!(tb_raw)  # ~1.1MB freed immediately
        out = m.mha(x_att, x_att; _attn_bias_flash=bias_flash)
    else
        # Original path for CPU or when mask is provided
        triangle_bias = m.linear(x_att) # (H, Q, B, batch_dim2)
        triangle_bias = permutedims(triangle_bias, (3, 1, 4, 2)) # (B, H, batch_dim2, Q)
        triangle_bias = reshape(triangle_bias, size(triangle_bias, 1), 1, size(triangle_bias, 2), size(triangle_bias, 3), size(triangle_bias, 4))

        if mask !== nothing
            if m.starting
                mask_att = permutedims(mask, (2, 3, 1)) # (J, B, I)
            else
                mask_att = permutedims(mask, (1, 3, 2)) # (I, B, J)
            end
            mask_bias = m.inf .* (mask_att .- 1)
            mask_bias = permutedims(mask_bias, (2, 3, 1))
            mask_bias = reshape(mask_bias, size(mask_bias, 1), size(mask_bias, 2), 1, 1, size(mask_bias, 3))
            biases = [mask_bias, triangle_bias]
        else
            biases = [triangle_bias]
        end
        out = m.mha(x_att, x_att; biases=biases)
    end

    if m.starting
        mha_result = out
        out = permutedims(mha_result, (1, 4, 2, 3)) # (C, I, J, B)
        isa(mha_result, CuArray) && CUDA.unsafe_free!(mha_result)
    else
        mha_result = out
        out = permutedims(mha_result, (1, 2, 4, 3)) # (C, I, J, B)
        isa(mha_result, CuArray) && CUDA.unsafe_free!(mha_result)
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
    linear_a_p = LinearFirst(c_z, c_hidden)
    linear_a_g = LinearFirst(c_z, c_hidden)
    linear_b_p = LinearFirst(c_z, c_hidden)
    linear_b_g = LinearFirst(c_z, c_hidden)
    linear_g = LinearFirst(c_z, c_z)
    linear_z = LinearFirst(c_hidden, c_z)
    layer_norm_in = LayerNormFirst(c_z)
    layer_norm_out = LayerNormFirst(c_hidden)
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
    # CUDA fast path: cuTENSOR contraction (no permutedims needed, zero-alloc with cached plans)
    if isa(a, CuArray)
        return cutensor_combine_projections(a, b, outgoing)
    elseif isa(a, SubArray) && isa(parent(a), CuArray)
        return cutensor_combine_projections_strided(a, b, outgoing)
    end

    # CPU fallback: permutedims + batched_mul
    # a, b: (C, L, L, B)
    a_perm = permutedims(a, (2, 3, 1, 4)) # (L, L, C, B)
    b_perm = permutedims(b, (2, 3, 1, 4)) # (L, L, C, B)
    L = size(a_perm, 1)
    C = size(a_perm, 3)
    B = size(a_perm, 4)
    a3 = reshape(a_perm, L, L, C * B)
    b3 = reshape(b_perm, L, L, C * B)
    if outgoing
        x3 = NNlib.batched_mul(a3, NNlib.batched_transpose(b3))
    else
        x3 = NNlib.batched_mul(NNlib.batched_transpose(a3), b3)
    end
    x2 = reshape(x3, L, L, C, B)
    x = permutedims(x2, (3, 1, 2, 4)) # (C, L, L, B)
    return x
end

function (m::TriangleMultiplicativeUpdate)(z::AbstractArray; mask=nothing)
    # z: (C_z, L, L, B)
    # CUDA fast path: copy + in-place LayerNorm to avoid allocating z_norm (~8.5MB)
    if isa(z, CuArray)
        z_norm = _get_perm_buf(42, size(z))
        copyto!(z_norm, z)
        layernorm_inplace!(m.layer_norm_in, z_norm)
    else
        z_norm = m.layer_norm_in(z)
    end

    # CUDA fast path: fuse 4 linear projections into 1 matmul (saves ~6 kernel launches)
    if isa(z_norm, CuArray)
        C = size(m.linear_a_p.weight, 1)
        fused = fused_linear_forward(
            (m.linear_a_g, m.linear_a_p, m.linear_b_g, m.linear_b_p), z_norm)
        # Compute gate projection while z_norm is still in L2 cache (pre-allocated output)
        g_raw = linear_forward!(43, m.linear_g, z_norm)
        # z_norm is pre-allocated buffer (slot 42), g_raw is slot 43 — no need to free
        # fused: (4C, L, L, B) — split via views and apply sigmoid+multiply
        trailing = ntuple(_ -> Colon(), ndims(fused) - 1)
        ag = @view fused[1:C, trailing...]
        ap = @view fused[C+1:2C, trailing...]
        bg = @view fused[2C+1:3C, trailing...]
        bp = @view fused[3C+1:4C, trailing...]
        # Write sigmoid*multiply results in-place into fused buffer views
        if mask !== nothing
            mask_r = reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
            @. ag = mask_r * NNlib.sigmoid(ag) * ap
            @. bg = mask_r * NNlib.sigmoid(bg) * bp
        else
            @. ag = NNlib.sigmoid(ag) * ap
            @. bg = NNlib.sigmoid(bg) * bp
        end
        a = ag  # view into fused[1:C, ...] with result
        b = bg  # view into fused[2C+1:3C, ...] with result

        x = _combine_projections(a, b, m.outgoing)
        # fused is a pre-allocated buffer (slot 40) — no need to free
        layernorm_inplace!(m.layer_norm_out, x)
        x = linear_forward!(44, m.linear_z, x)
        @. x = x * NNlib.sigmoid(g_raw)
        # x and g_raw are pre-allocated buffers — no need to free
        return x
    end

    # CPU fallback
    a_g = m.linear_a_g(z_norm)
    a_p = m.linear_a_p(z_norm)
    b_g = m.linear_b_g(z_norm)
    b_p = m.linear_b_p(z_norm)
    if mask !== nothing
        mask_r = reshape(mask, 1, size(mask, 1), size(mask, 2), size(mask, 3))
        a = @. mask_r * NNlib.sigmoid(a_g) * a_p
        b = @. mask_r * NNlib.sigmoid(b_g) * b_p
    else
        a = @. NNlib.sigmoid(a_g) * a_p
        b = @. NNlib.sigmoid(b_g) * b_p
    end

    x = _combine_projections(a, b, m.outgoing)
    x = m.layer_norm_out(x)
    x = m.linear_z(x)
    g_raw = m.linear_g(z_norm)
    @. x = x * NNlib.sigmoid(g_raw)
    return x
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

    layernorm_1 = LayerNormFirst(sequence_state_dim)
    sequence_to_pair = SequenceToPair(sequence_state_dim, pairwise_state_dim ÷ 2, pairwise_state_dim)
    pair_to_sequence = PairToSequence(pairwise_state_dim, sequence_num_heads)

    seq_attention = ESMFoldAttention(sequence_state_dim, sequence_num_heads, sequence_head_width; gated=true)

    tri_mul_out = TriangleMultiplicationOutgoing(pairwise_state_dim, pairwise_state_dim)
    tri_mul_in = TriangleMultiplicationIncoming(pairwise_state_dim, pairwise_state_dim)
    tri_att_start = TriangleAttention(pairwise_state_dim, pairwise_head_width, pairwise_num_heads; starting=true, inf=1e9)
    tri_att_end = TriangleAttention(pairwise_state_dim, pairwise_head_width, pairwise_num_heads; starting=false, inf=1e9)

    mlp_seq = ResidueMLP(sequence_state_dim, 4 * sequence_state_dim; dropout=dropout)
    mlp_pair = ResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim; dropout=dropout)

    drop = SharedDropout(dropout, 3)
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
    # sequence_state: (C_s, L, B), pairwise_state: (C_z, L, L, B)
    bias = m.pair_to_sequence(pairwise_state) # (H, L, L, B)
    y = m.layernorm_1(sequence_state)
    y, _ = m.seq_attention(y; mask=mask, bias=bias)
    isa(bias, CuArray) && CUDA.unsafe_free!(bias)  # free bias after attention

    # CUDA fast path: in-place residual additions to avoid allocating new arrays
    if isa(sequence_state, CuArray)
        attn_out = m.drop(y)
        sequence_state .+= attn_out
        CUDA.unsafe_free!(attn_out)
    else
        sequence_state = sequence_state .+ m.drop(y)
    end
    sequence_state = m.mlp_seq(sequence_state)

    if isa(pairwise_state, CuArray)
        s2p_out = m.sequence_to_pair(sequence_state)
        pairwise_state .+= s2p_out
        CUDA.unsafe_free!(s2p_out)

        tri_mask = mask === nothing ? nothing :
            (reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2)))
        # tri_mul outputs are pre-allocated buffers — no need to free
        pairwise_state .+= m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
        pairwise_state .+= m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
        # tri_att outputs are fresh allocations — free after use
        tmp = m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
        pairwise_state .+= tmp
        CUDA.unsafe_free!(tmp)
        tmp = m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))
        pairwise_state .+= tmp
        CUDA.unsafe_free!(tmp)
    else
        pairwise_state = pairwise_state .+ m.sequence_to_pair(sequence_state)

        tri_mask = mask === nothing ? nothing :
            (reshape(mask, size(mask, 1), 1, size(mask, 2)) .* reshape(mask, 1, size(mask, 1), size(mask, 2)))
        pairwise_state = pairwise_state .+ m.row_drop(m.tri_mul_out(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.col_drop(m.tri_mul_in(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.row_drop(m.tri_att_start(pairwise_state; mask=tri_mask))
        pairwise_state = pairwise_state .+ m.col_drop(m.tri_att_end(pairwise_state; mask=tri_mask))
    end

    pairwise_state = m.mlp_pair(pairwise_state)

    return sequence_state, pairwise_state
end
