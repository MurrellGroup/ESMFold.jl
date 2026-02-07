# GPU-optimized kernels for ESMFold.jl
# Uses cuTile.jl for fused flash attention and other optimizations.
# CUDA and cuTile are already imported by the parent module.

const _INV_LOG_2 = Float32(1 / log(2))
const _ConstInt = ct.Constant{Int}
const _ConstBool = ct.Constant{Bool}

# ============================================================================
# TF32 Math Mode
# ============================================================================

"""
    enable_tf32!()

Enable TF32 math mode for cuBLAS matmuls (~2x Tensor Core speedup).
cuTile kernels use TF32 natively via ct.TFloat32 conversion.
"""
function enable_tf32!()
    CUDA.math_mode!(CUDA.FAST_MATH)
end

# Helper: convert tile to TFloat32 for tensor core acceleration on Float32 data
@inline _to_tf32(tile, ::Type{Float32}) = convert(ct.Tile{ct.TFloat32}, tile)
@inline _to_tf32(tile, ::Type{T}) where T = tile  # no-op for Float16/BFloat16

# ============================================================================
# Flash Multi-Head Attention (cuTile kernel)
# ============================================================================

# Layout: (D, SeqLen, Heads, Batch) - matches ESMFold conventions after reshape
function _fmha_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Out::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)

    m_i = ct.full((1, TILE_M[]), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M[]), Float32)
    acc = ct.zeros((D_V[], TILE_M[]), Float32)

    q = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
    q = reshape(q, (D_K[], TILE_M[]))

    k_seqlen = K.sizes[2]
    Tc = cld(k_seqlen, TILE_N[])

    j = Int32(1)
    while j <= Tc
        k = ct.load(K, (1, j, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1), latency=2)
        k = reshape(k, (D_K[], TILE_N[]))
        k = transpose(k)

        # Q*K^T with TF32 tensor cores for Float32 data
        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k, T), _to_tf32(q, T), qk)
        else
            qk = ct.muladd(k, q, qk)
        end

        # Mask out-of-bounds K positions to -Inf (only on last tile when not aligned).
        # ct.load zero-pads OOB, giving QK=0 → exp2(0)=1 which corrupts softmax.
        if j == Tc
            k_idx = (j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        # Online softmax
        m_ij = max.(m_i, maximum(qk, dims=1) * qk_scale_log2)
        qk = qk * qk_scale_log2 .- m_ij
        p = exp2.(qk)
        l_ij = sum(p, dims=1)
        alpha = exp2.(m_i .- m_ij)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        # V accumulation with TF32 tensor cores
        v = ct.load(V, (1, j, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1), latency=4)
        v = reshape(v, (D_V[], TILE_N[]))
        p = ct.astype(p, T)
        if T === Float32
            acc = ct.muladd(_to_tf32(v, T), _to_tf32(p, T), acc)
        else
            acc = ct.muladd(v, p, acc)
        end
        m_i = m_ij

        j += Int32(1)
    end

    acc = acc ./ l_i
    acc = reshape(acc, (D_V[], TILE_M[], 1, 1))
    acc = ct.astype(acc, T)
    ct.store(Out, (1, bid_x, head_idx, batch_idx), acc)

    return
end

# Flash attention with additive bias support (for ESMFoldAttention and OFMultiheadAttention)
function _fmha_bias_kernel(
    Q::ct.TileArray{T,4},
    K::ct.TileArray{T,4},
    V::ct.TileArray{T,4},
    Bias::ct.TileArray{T,4},
    Out::ct.TileArray{T,4},
    qk_scale::AbstractFloat,
    D_K::_ConstInt,
    D_V::_ConstInt,
    H::_ConstInt,
    TILE_M::_ConstInt,
    TILE_N::_ConstInt,
    BIAS_BATCH::_ConstInt,
) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    batch_idx, head_idx = fldmod1(bid_y, H[])

    # Broadcast bias: when BIAS_BATCH < actual batch, use modular indexing
    bias_batch_idx = BIAS_BATCH[] == Int32(1) ? Int32(1) : batch_idx

    qk_scale_log2 = Float32(qk_scale) * Float32(_INV_LOG_2)

    m_i = ct.full((1, TILE_M[]), -Inf32, Float32)
    l_i = ct.zeros((1, TILE_M[]), Float32)
    acc = ct.zeros((D_V[], TILE_M[]), Float32)

    q = ct.load(Q, (1, bid_x, head_idx, batch_idx), (D_K[], TILE_M[], 1, 1))
    q = reshape(q, (D_K[], TILE_M[]))

    k_seqlen = K.sizes[2]
    Tc = cld(k_seqlen, TILE_N[])

    j = Int32(1)
    while j <= Tc
        k = ct.load(K, (1, j, head_idx, batch_idx), (D_K[], TILE_N[], 1, 1), latency=2)
        k = reshape(k, (D_K[], TILE_N[]))
        k = transpose(k)

        # Q*K^T with TF32 tensor cores
        qk = ct.zeros((TILE_N[], TILE_M[]), Float32)
        if T === Float32
            qk = ct.muladd(_to_tf32(k, T), _to_tf32(q, T), qk)
        else
            qk = ct.muladd(k, q, qk)
        end

        # Scale QK to log2 space, then add bias (also converted to log2 space)
        qk = qk * qk_scale_log2

        # Load and add bias tile: Bias layout is (SeqK, SeqQ, H, bias_batch)
        bias_tile = ct.load(Bias, (j, bid_x, head_idx, bias_batch_idx), (TILE_N[], TILE_M[], 1, 1))
        bias_tile = reshape(bias_tile, (TILE_N[], TILE_M[]))
        qk = qk .+ bias_tile * Float32(_INV_LOG_2)

        # Mask out-of-bounds K positions to -Inf (only on last tile when not aligned).
        if j == Tc
            k_idx = (j - Int32(1)) * Int32(TILE_N[]) .+ ct.arange((TILE_N[],), Int32)
            k_penalty = ifelse.(k_idx .<= Int32(k_seqlen), 0f0, -Inf32)
            qk = qk .+ reshape(k_penalty, (TILE_N[], 1))
        end

        # Online softmax (already in log2 space)
        m_ij = max.(m_i, maximum(qk, dims=1))
        qk = qk .- m_ij
        p = exp2.(qk)
        l_ij = sum(p, dims=1)
        alpha = exp2.(m_i .- m_ij)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* alpha

        # V accumulation with TF32 tensor cores
        v = ct.load(V, (1, j, head_idx, batch_idx), (D_V[], TILE_N[], 1, 1), latency=4)
        v = reshape(v, (D_V[], TILE_N[]))
        p = ct.astype(p, T)
        if T === Float32
            acc = ct.muladd(_to_tf32(v, T), _to_tf32(p, T), acc)
        else
            acc = ct.muladd(v, p, acc)
        end
        m_i = m_ij

        j += Int32(1)
    end

    acc = acc ./ l_i
    acc = reshape(acc, (D_V[], TILE_M[], 1, 1))
    acc = ct.astype(acc, T)
    ct.store(Out, (1, bid_x, head_idx, batch_idx), acc)

    return
end

# ============================================================================
# Fused Rotary Positional Embedding (CuArray fast path)
# ============================================================================

"""
    apply_rotary_pos_emb_fused(x::CuArray, cos, sin)

Fused rotary positional embedding using 2 broadcast kernels + 1 vcat
instead of ~5 separate kernels. Works for any ndims >= 2.
cos, sin: (d, seq_len_cached) pre-computed on device.
"""
function apply_rotary_pos_emb_fused(x::CuArray{T}, cos::AbstractArray, sin::AbstractArray) where T
    d = size(x, 1)
    seq_len = size(x, 2)
    half = d ÷ 2
    trailing = ntuple(_ -> Colon(), ndims(x) - 1)

    x1 = @view x[1:half, trailing...]
    x2 = @view x[half+1:d, trailing...]

    # cos/sin views: (half, seq_len) — broadcasts against trailing dims
    c1 = @view cos[1:half, 1:seq_len]
    c2 = @view cos[half+1:d, 1:seq_len]
    s1 = @view sin[1:half, 1:seq_len]
    s2 = @view sin[half+1:d, 1:seq_len]

    # Pre-allocate output and write both halves via fused broadcasts (2 GPU kernels)
    out = similar(x)
    o1 = @view out[1:half, trailing...]
    o2 = @view out[half+1:d, trailing...]

    @. o1 = x1 * c1 - x2 * s1
    @. o2 = x2 * c2 + x1 * s2

    return out
end

# ============================================================================
# Fused Multi-Linear Projection (batch multiple LinearFirst into single matmul)
# ============================================================================

"""
    _fused_linear_cache - Global cache for fused weight/bias matrices.
    Keyed by tuple of objectid(weight) for constituent layers.
"""
const _fused_linear_cache = Dict{UInt, Tuple{Any, Any}}()

"""
    _get_fused_weights(layers::Tuple) -> (fused_weight, fused_bias)

Retrieve or create fused weight/bias by concatenating multiple LinearFirst layers.
Returns (fused_weight, fused_bias) where fused_weight is (sum_out_dims, in_dim)
and fused_bias is (sum_out_dims,).
"""
function _get_fused_weights(layers::Tuple)
    key = hash(map(l -> objectid(l.weight), layers))
    if !haskey(_fused_linear_cache, key)
        fused_w = cat(map(l -> l.weight, layers)...; dims=1)
        fused_b = cat(map(l -> l.use_bias ? l.bias : zeros_like(l.weight, eltype(l.weight), size(l.weight, 1)), layers)...; dims=1)
        _fused_linear_cache[key] = (fused_w, fused_b)
    end
    return _fused_linear_cache[key]
end

"""
    fused_linear_forward(layers::Tuple, x::CuArray) -> fused_output

Apply multiple LinearFirst layers as a single fused matmul.
Returns the raw concatenated output (sum_out_dims, ...) — caller splits via views.
Uses pre-allocated output buffer to avoid 34MB allocation per call.
"""
function fused_linear_forward(layers::Tuple, x::CuArray)
    fused_w, fused_b = _get_fused_weights(layers)
    in_dim = size(fused_w, 2)
    out_dim = size(fused_w, 1)
    x2 = reshape(x, in_dim, :)
    n_cols = size(x2, 2)
    # Pre-allocated output buffer (avoids ~34MB allocation per call)
    y2 = _get_perm_buf(40, (out_dim, n_cols))
    LinearAlgebra.mul!(y2, fused_w, x2)
    y2 .+= fused_b
    trailing = size(x)[2:end]
    return reshape(y2, out_dim, trailing...)
end

"""
    linear_forward!(slot, m::LinearFirst, x::CuArray) -> pre-allocated output

Apply a LinearFirst layer using a pre-allocated output buffer (via mul!).
Avoids allocation for the matmul result.
"""
function linear_forward!(slot::Int, m, x::CuArray)
    in_dim = size(m.weight, 2)
    out_dim = size(m.weight, 1)
    trailing = size(x)[2:end]
    x2 = reshape(x, in_dim, :)
    n_cols = size(x2, 2)
    y2 = _get_perm_buf(slot, (out_dim, n_cols))
    LinearAlgebra.mul!(y2, m.weight, x2)
    if m.use_bias
        y2 .+= m.bias
    end
    return reshape(y2, out_dim, trailing...)
end

"""
    _select_fmha_tiles(D_k, seq_len, heads, batch) -> (tile_m, tile_n)

Auto-select flash attention tile sizes based on problem dimensions.
Smaller D_k allows smaller tile_m for better occupancy with large batches.
"""
function _select_fmha_tiles(D_k::Int, seq_len::Int, heads::Int, batch::Int)
    if D_k <= 32 && batch * heads >= 64
        return (32, 64)  # More blocks for better SM utilization with small head dim
    else
        return (64, 64)   # Default: good balance for D_k=64 cases
    end
end

"""
    flash_attention(Q, K, V; scale=nothing) -> Out

Fused multi-head attention using cuTile with TF32 tensor cores.
Q, K, V: (D, SeqLen, Heads, Batch)
Returns: (D, SeqLen, Heads, Batch)
"""
function flash_attention(
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4};
    scale::Union{Nothing,Real}=nothing,
    tile_m::Int=-1, tile_n::Int=-1,
    output::Union{Nothing,CuArray{T,4}}=nothing,
) where T
    D_k, seq_q, heads, batch = size(Q)
    D_v = size(V, 1)

    if tile_m == -1
        tile_m, tile_n = _select_fmha_tiles(D_k, seq_q, heads, batch)
    end

    qk_scale = something(scale, 1f0 / sqrt(Float32(D_k)))

    Out = output === nothing ? similar(V, T, D_v, seq_q, heads, batch) : output

    grid_x = cld(seq_q, tile_m)
    grid_y = heads * batch
    grid = (grid_x, grid_y)

    ct.launch(_fmha_kernel, grid, Q, K, V, Out,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n))

    return Out
end

"""
    flash_attention_bias(Q, K, V, bias; scale=nothing) -> Out

Fused multi-head attention with additive bias using cuTile with TF32 tensor cores.
Q, K, V: (D, SeqLen, Heads, Batch)
bias: (SeqK, SeqQ, Heads, Batch) - additive attention bias
Returns: (D, SeqLen, Heads, Batch)
"""
function flash_attention_bias(
    Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4},
    bias::CuArray{<:Real,4};
    scale::Union{Nothing,Real}=nothing,
    tile_m::Int=-1, tile_n::Int=-1,
    output::Union{Nothing,CuArray{T,4}}=nothing,
) where T
    D_k, seq_q, heads, batch = size(Q)
    D_v = size(V, 1)
    bias_batch = size(bias, 4)  # may be < batch for broadcast

    if tile_m == -1
        tile_m, tile_n = _select_fmha_tiles(D_k, seq_q, heads, batch)
    end

    qk_scale = something(scale, 1f0 / sqrt(Float32(D_k)))

    Out = output === nothing ? similar(V, T, D_v, seq_q, heads, batch) : output
    bias_t = eltype(bias) === T ? bias : T.(bias)  # skip copy if already correct type

    grid_x = cld(seq_q, tile_m)
    grid_y = heads * batch
    grid = (grid_x, grid_y)

    ct.launch(_fmha_bias_kernel, grid, Q, K, V, bias_t, Out,
        qk_scale,
        ct.Constant(D_k), ct.Constant(D_v), ct.Constant(heads),
        ct.Constant(tile_m), ct.Constant(tile_n),
        ct.Constant(bias_batch))

    return Out
end

# ============================================================================
# Optimized ESM2 Multi-Head Attention (flash attention dispatch)
# ============================================================================

"""
    esm2_flash_attention(q, k, v, num_heads, batch)

Replaces the batched_mul -> softmax -> batched_mul pattern in ESMMultiheadAttention.
Inputs q, k, v are already projected and scaled: (head_dim, seq_len, batch * num_heads)
"""
function esm2_flash_attention(
    q::CuArray{T,3}, k::CuArray{T,3}, v::CuArray{T,3},
    num_heads::Int, batch::Int,
) where T
    head_dim, seq_len, bh = size(q)
    @assert bh == batch * num_heads

    # Reshape to 4D: (D, SeqLen, Heads, Batch)
    q4 = reshape(q, head_dim, seq_len, num_heads, batch)
    k4 = reshape(k, head_dim, seq_len, num_heads, batch)
    v4 = reshape(v, head_dim, seq_len, num_heads, batch)

    # Scale is already applied to q, so use scale=1.0
    out4 = flash_attention(q4, k4, v4; scale=1.0f0)

    # Reshape back to (D, SeqLen, B*H)
    return reshape(out4, head_dim, seq_len, bh)
end

# ============================================================================
# Optimized ESMFold Attention (flash attention dispatch)
# ============================================================================

"""
    esmfold_flash_attention(q3, k3, v3, num_heads, batch; bias_3d=nothing)

Flash attention for ESMFoldAttention with bias support.
q3, k3, v3: (L, D, B*H) -> needs transpose to (D, L, H, B)
bias_3d: (L, L, B*H) or nothing
"""
function esmfold_flash_attention(
    q3::CuArray{T,3}, k3::CuArray{T,3}, v3::CuArray{T,3},
    num_heads::Int, batch::Int;
    bias_3d::Union{CuArray,Nothing}=nothing,
) where T
    L, D, BH = size(q3)

    # Transpose from (L, D, B*H) to (D, L, H, B)
    q4 = reshape(permutedims(q3, (2, 1, 3)), D, L, num_heads, batch)
    k4 = reshape(permutedims(k3, (2, 1, 3)), D, L, num_heads, batch)
    v4 = reshape(permutedims(v3, (2, 1, 3)), D, L, num_heads, batch)

    if bias_3d !== nothing
        # bias_3d: (L, L, B*H) -> (L, L, H, B)
        bias4 = reshape(bias_3d, L, L, num_heads, batch)
        out4 = flash_attention_bias(q4, k4, v4, bias4; scale=1.0f0)
    else
        out4 = flash_attention(q4, k4, v4; scale=1.0f0)
    end

    # Back to (L, D, B*H)
    out3 = reshape(out4, D, L, BH)
    return permutedims(out3, (2, 1, 3))
end

# ============================================================================
# Fused LayerNorm (cuTile kernel) — single kernel, zero intermediate allocations
# ============================================================================

function _layernorm_first_kernel(
    X::ct.TileArray{T,2},
    Out::ct.TileArray{T,2},
    W::ct.TileArray{T,2},
    B_arr::ct.TileArray{T,2},
    eps::AbstractFloat,
    C_DIM::_ConstInt,
    TILE_N::_ConstInt,
) where T
    bid = ct.bid(1)

    # Load input tile: (C, TILE_N) — one column = one normalization group
    x = ct.load(X, (1, bid), (C_DIM[], TILE_N[]))

    # Load weight and bias: (C, 1) — broadcast across N positions
    w = ct.load(W, (1, 1), (C_DIM[], 1))
    b = ct.load(B_arr, (1, 1), (C_DIM[], 1))

    # Mean: reduce over dim 1 (the channel dimension)
    inv_c = Float32(1.0 / C_DIM[])
    μ = sum(x, dims=1) * inv_c  # (1, TILE_N)

    # Variance and normalize
    diff = x .- μ
    σ2 = sum(diff .* diff, dims=1) * inv_c  # (1, TILE_N)
    inv_std = 1f0 ./ sqrt.(σ2 .+ Float32(eps))

    # Scale + shift: w and b have shape (C, 1), broadcast across TILE_N
    out = @. diff * inv_std * w + b

    ct.store(Out, (1, bid), out)
    return
end

"""
    fused_layernorm_first(x::CuArray, w, b, eps) -> normalized output

Fused LayerNorm using a cuTile kernel: 1 kernel launch, 0 intermediate allocations.
Replaces the ~9 kernel launches + 6 intermediate allocations of the non-fused version.
"""
function fused_layernorm_first(x::CuArray{T}, w_vec::CuArray{T}, b_vec::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    orig_size = size(x)
    N = div(length(x), C)

    x2 = reshape(x, C, N)
    out2 = similar(x2)

    # Reshape weight/bias to (C, 1) for tile loading
    w2 = reshape(w_vec, C, 1)
    b2 = reshape(b_vec, C, 1)

    # Choose tile_n based on C to keep register usage reasonable
    tile_n = C <= 256 ? 64 : (C <= 1024 ? 16 : 8)

    grid = (cld(N, tile_n),)

    ct.launch(_layernorm_first_kernel, grid, x2, out2, w2, b2,
        eps, ct.Constant(C), ct.Constant(tile_n))

    return reshape(out2, orig_size)
end

"""
    fused_layernorm_first!(x::CuArray, w, b, eps) -> x (modified in-place)

In-place variant of fused LayerNorm. Safe because each cuTile thread block
processes its own tile independently (load → compute → store, no cross-block deps).
Saves one allocation (~8.5 MB for pairwise state tensors).
"""
function fused_layernorm_first!(x::CuArray{T}, w_vec::CuArray{T}, b_vec::CuArray{T}, eps::Float32) where T
    C = size(x, 1)
    N = div(length(x), C)

    x2 = reshape(x, C, N)
    w2 = reshape(w_vec, C, 1)
    b2 = reshape(b_vec, C, 1)

    tile_n = C <= 256 ? 64 : (C <= 1024 ? 16 : 8)
    grid = (cld(N, tile_n),)

    ct.launch(_layernorm_first_kernel, grid, x2, x2, w2, b2,
        eps, ct.Constant(C), ct.Constant(tile_n))

    return x
end

# ============================================================================
# Pre-allocated buffer pool for permutedims! (avoids allocation per call)
# ============================================================================

"""
    _perm_buf_pool - Global cache for pre-allocated CuArray buffers used with permutedims!.
    Keyed by (slot_id, shape) to allow multiple buffers of the same shape.
"""
const _perm_buf_pool = Dict{Tuple{Int, Tuple}, CuArray}()

function _get_perm_buf(slot::Int, shape::Tuple)
    key = (slot, shape)
    if !haskey(_perm_buf_pool, key)
        _perm_buf_pool[key] = CUDA.zeros(Float32, shape...)
    end
    return _perm_buf_pool[key]
end

# ============================================================================
# cuTENSOR-accelerated _combine_projections
# ============================================================================

"""
    _cutensor_plan_cache - Global cache for cuTENSOR contraction plans + buffers.
    Keyed by (C, L, B, outgoing) for each unique tensor shape/mode.
"""
const _cutensor_plan_cache = Dict{Tuple{Int,Int,Int,Bool}, Any}()

"""
    _get_cutensor_plan(C, L, B, outgoing) -> (plan, a_buf, b_buf, result_buf)

Get or create a cached cuTENSOR contraction plan and pre-allocated buffers.
"""
function _get_cutensor_plan(C::Int, L::Int, B::Int, outgoing::Bool)
    key = (C, L, B, outgoing)
    if !haskey(_cutensor_plan_cache, key)
        a_buf = CUDA.zeros(Float32, C, L, L, B)
        b_buf = CUDA.zeros(Float32, C, L, L, B)
        result_buf = CUDA.zeros(Float32, C, L, L, B)

        if outgoing
            plan = cuTENSOR.plan_contraction(
                a_buf, Int32[1, 2, 3, 4], cuTENSOR.OP_IDENTITY,
                b_buf, Int32[1, 5, 3, 4], cuTENSOR.OP_IDENTITY,
                result_buf, Int32[1, 2, 5, 4], cuTENSOR.OP_IDENTITY,
                cuTENSOR.OP_IDENTITY)
        else
            plan = cuTENSOR.plan_contraction(
                a_buf, Int32[1, 3, 2, 4], cuTENSOR.OP_IDENTITY,
                b_buf, Int32[1, 3, 5, 4], cuTENSOR.OP_IDENTITY,
                result_buf, Int32[1, 2, 5, 4], cuTENSOR.OP_IDENTITY,
                cuTENSOR.OP_IDENTITY)
        end

        _cutensor_plan_cache[key] = (plan, a_buf, b_buf, result_buf)
    end
    return _cutensor_plan_cache[key]
end

function cutensor_combine_projections(a::CuArray, b::CuArray, outgoing::Bool)
    C, L1, L2, B = size(a)
    plan, a_buf, b_buf, result_buf = _get_cutensor_plan(C, L1, B, outgoing)
    copyto!(a_buf, a)
    copyto!(b_buf, b)
    cuTENSOR.contract!(plan, 1f0, a_buf, b_buf, 0f0, result_buf)
    return result_buf
end

function cutensor_combine_projections_strided(a, b, outgoing::Bool)
    C = size(a, 1)
    L = size(a, 2)
    B = size(a, 4)
    plan, a_buf, b_buf, result_buf = _get_cutensor_plan(C, L, B, outgoing)
    copyto!(a_buf, a)
    copyto!(b_buf, b)
    cuTENSOR.contract!(plan, 1f0, a_buf, b_buf, 0f0, result_buf)
    return result_buf
end

