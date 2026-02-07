using NNlib
using Statistics

# ESMFold misc utilities

@concrete struct ESMFoldAttention <: Onion.Layer
    proj
    o_proj
    g_proj
    num_heads::Int
    head_width::Int
    gated::Bool
    rescale_factor::Float32
end

@layer ESMFoldAttention

function ESMFoldAttention(embed_dim::Int, num_heads::Int, head_width::Int; gated::Bool=false)
    proj = LinearFirst(embed_dim, embed_dim * 3; bias=false)
    o_proj = LinearFirst(embed_dim, embed_dim; bias=true)
    g_proj = gated ? LinearFirst(embed_dim, embed_dim; bias=true) : nothing
    rescale_factor = Float32(head_width)^-0.5f0
    return ESMFoldAttention(proj, o_proj, g_proj, num_heads, head_width, gated, rescale_factor)
end

function (m::ESMFoldAttention)(x::AbstractArray; mask=nothing, bias=nothing)
    # x: (C, L, B)
    t = m.proj(x)
    D = m.head_width
    H = m.num_heads
    L = size(t, 2)
    B = size(t, 3)
    # Reshape to (head_width, 3, num_heads, L, B)
    t = reshape(t, D, 3, H, L, B)

    # Fast CUDA path: extract q/k/v directly into flash attention format (D, L, H, B)
    if isa(t, CuArray)
        qkv_shape = (D, L, H, B)
        q4 = _get_perm_buf(20, qkv_shape)
        k4 = _get_perm_buf(21, qkv_shape)
        v4 = _get_perm_buf(22, qkv_shape)
        permutedims!(q4, @view(t[:, 1, :, :, :]), (1, 3, 2, 4))  # (D, L, H, B)
        permutedims!(k4, @view(t[:, 2, :, :, :]), (1, 3, 2, 4))
        permutedims!(v4, @view(t[:, 3, :, :, :]), (1, 3, 2, 4))
        CUDA.unsafe_free!(t)  # ~6MB freed (D*3*H*L*B)

        # Build bias for flash attention: need (K, Q, H, B) format
        bias4 = nothing
        if bias !== nothing
            bias4 = permutedims(bias, (3, 2, 1, 4))
        end
        if mask !== nothing
            mask_bias = reshape(ifelse.(mask .== 1, 0f0, Float32(-Inf)), L, 1, 1, B)
            if bias4 !== nothing
                bias4 = bias4 .+ mask_bias
            else
                bias4 = repeat(mask_bias, 1, L, H, 1)
            end
        end

        flash_out = _get_perm_buf(23, qkv_shape)
        if bias4 !== nothing
            flash_attention_bias(q4, k4, v4, bias4; scale=m.rescale_factor, output=flash_out)
        else
            flash_attention(q4, k4, v4; scale=m.rescale_factor, output=flash_out)
        end

        # Output: (D, L, H, B) → (D, H, L, B) → reshape to (C, L, B)
        out_perm = _get_perm_buf(24, (D, H, L, B))
        permutedims!(out_perm, flash_out, (1, 3, 2, 4))
        o = reshape(out_perm, D * H, L, B)

        if m.gated
            g_raw = m.g_proj(x)
            @. o = NNlib.sigmoid(g_raw) * o
            CUDA.unsafe_free!(g_raw)  # free gating projection immediately
        end

        o_result = m.o_proj(o)
        return o_result, nothing
    end

    # CPU fallback: standard batched_mul path
    t = permutedims(t, (5, 3, 4, 2, 1))  # (B, H, L, 3, D)
    q = view(t, :, :, :, 1, :)
    k = view(t, :, :, :, 2, :)
    v = view(t, :, :, :, 3, :)
    q = q .* m.rescale_factor

    q3 = permutedims(reshape(q, B * H, L, D), (2, 3, 1))
    k3 = permutedims(reshape(k, B * H, L, D), (2, 3, 1))
    v3 = permutedims(reshape(v, B * H, L, D), (2, 3, 1))

    a = NNlib.batched_mul(q3, permutedims(k3, (2, 1, 3)))

    if bias !== nothing
        bias_h = permutedims(bias, (2, 3, 4, 1))
        b3 = reshape(bias_h, size(bias_h, 1), size(bias_h, 2), :)
        a = a .+ b3
    end

    if mask !== nothing
        mask_b = permutedims(mask, (2, 1))
        mask_k = reshape(mask_b, B, 1, 1, L)
        mask_k = repeat(mask_k, 1, H, L, 1)
        neg_inf = oftype(zero(eltype(a)), -Inf)
        mask_bias = ifelse.(mask_k .== 1, zero(eltype(a)), neg_inf)
        mb_perm = permutedims(mask_bias, (3, 4, 1, 2))
        mb3 = reshape(mb_perm, size(mb_perm, 1), size(mb_perm, 2), :)
        a = a .+ mb3
    end

    a = NNlib.softmax(a; dims=2)
    o = NNlib.batched_mul(a, v3)

    o = reshape(o, L, D, B, H)
    o = permutedims(o, (2, 4, 1, 3))
    o = reshape(o, D * H, L, B)

    if m.gated
        g = NNlib.sigmoid.(m.g_proj(x))
        o = g .* o
    end

    return m.o_proj(o), nothing
end

# Training-mode toggle for dropout-like layers.
const _TRAINING = Ref(false)

set_training!(flag::Bool) = (_TRAINING[] = flag)
is_training() = _TRAINING[]

@concrete struct SharedDropout <: Onion.Layer
    r::Float32
    batch_dim::Vector{Int}
end

@layer SharedDropout

function SharedDropout(r::Real, batch_dim)
    bd = isa(batch_dim, Int) ? [batch_dim] : collect(batch_dim)
    return SharedDropout(Float32(r), bd)
end

function (m::SharedDropout)(x)
    (m.r == 0f0 || !_TRAINING[]) && return x
    shape = collect(size(x))
    for bd in m.batch_dim
        shape[bd] = 1
    end
    mask = rand(eltype(x), shape...)
    keep = one(eltype(x)) - eltype(x)(m.r)
    scale = one(eltype(x)) / keep
    mask = ifelse.(mask .>= eltype(x)(m.r), scale, zero(eltype(x)))
    return x .* mask
end

@concrete struct SequenceToPair <: Onion.Layer
    layernorm
    proj
    o_proj
end

@layer SequenceToPair

function SequenceToPair(sequence_state_dim::Int, inner_dim::Int, pairwise_state_dim::Int)
    layernorm = LayerNormFirst(sequence_state_dim)
    proj = LinearFirst(sequence_state_dim, inner_dim * 2)
    o_proj = LinearFirst(2 * inner_dim, pairwise_state_dim)
    return SequenceToPair(layernorm, proj, o_proj)
end

function (m::SequenceToPair)(sequence_state)
    # sequence_state: (C_s, L, B)
    s_norm = m.layernorm(sequence_state)
    s = m.proj(s_norm)
    isa(s_norm, CuArray) && CUDA.unsafe_free!(s_norm)  # free LayerNorm output
    inner_dim = size(s, 1) ÷ 2
    q = view(s, 1:inner_dim, :, :)
    k = view(s, (inner_dim + 1):(2 * inner_dim), :, :)
    L = size(q, 2)
    B = size(q, 3)
    q_exp = reshape(q, inner_dim, 1, L, B)
    k_exp = reshape(k, inner_dim, L, 1, B)

    if isa(s, CuArray)
        # Pre-allocated outer product buffer (avoids ~8.5MB allocation per call)
        x = _get_perm_buf(30, (2 * inner_dim, L, L, B))
        prod_view = @view x[1:inner_dim, :, :, :]
        diff_view = @view x[inner_dim+1:end, :, :, :]
        @. prod_view = q_exp * k_exp
        @. diff_view = q_exp - k_exp
        CUDA.unsafe_free!(s)  # free proj output after outer product
    else
        prod = q_exp .* k_exp
        diff = q_exp .- k_exp
        x = cat(prod, diff; dims=1)
    end

    x = m.o_proj(x)
    return x
end

@concrete struct PairToSequence <: Onion.Layer
    layernorm
    linear
end

@layer PairToSequence

function PairToSequence(pairwise_state_dim::Int, num_heads::Int)
    layernorm = LayerNormFirst(pairwise_state_dim)
    linear = LinearFirst(pairwise_state_dim, num_heads; bias=false)
    return PairToSequence(layernorm, linear)
end

function (m::PairToSequence)(pairwise_state)
    # pairwise_state: (C_z, L, L, B)
    z = m.layernorm(pairwise_state)
    result = m.linear(z)
    isa(z, CuArray) && CUDA.unsafe_free!(z)  # free LayerNorm output
    return result
end

@concrete struct ResidueMLP <: Onion.Layer
    norm
    fc1
    fc2
    dropout
end

@layer ResidueMLP

function ResidueMLP(embed_dim::Int, inner_dim::Int; dropout::Real=0)
    norm = LayerNormFirst(embed_dim)
    fc1 = LinearFirst(embed_dim, inner_dim)
    fc2 = LinearFirst(inner_dim, embed_dim)
    drop = SharedDropout(dropout, 3)
    return ResidueMLP(norm, fc1, fc2, drop)
end

function (m::ResidueMLP)(x)
    y = m.norm(x)
    norm_out = y  # keep reference for freeing
    y = m.fc1(y)
    if isa(x, CuArray)
        CUDA.unsafe_free!(norm_out)  # free LayerNorm output after fc1 consumed it
    end
    y .= max.(y, 0f0)  # in-place ReLU: reuse fc1's output buffer (~47MB saved for mlp_pair)
    fc1_out = y  # keep reference for freeing
    y = m.fc2(y)
    if isa(x, CuArray)
        CUDA.unsafe_free!(fc1_out)  # free fc1 output after fc2 consumed it
    end
    y = m.dropout(y)
    # In-place residual addition on CUDA to avoid allocating a new array
    if isa(x, CuArray)
        x .+= y
        CUDA.unsafe_free!(y)  # free fc2 output after addition
        return x
    end
    return x .+ y
end

# Sequence encoding utilities

function encode_sequence(
    seq::AbstractString;
    residue_index_offset::Int=512,
    chain_linker::Union{AbstractString,Int}="G"^25,
)
    chains = split(seq, ":")
    full_seq = chain_linker isa AbstractString ? join(chains, chain_linker) : join(chains, "")
    unk_idx = restype_order_with_x["X"]
    encoded = [get(restype_order_with_x, string(aa), unk_idx) for aa in full_seq]
    residx = Int[]
    linker_mask = ones(Float32, length(encoded))
    chain_index = Int[]

    if chain_linker isa AbstractString
        residx = collect(0:(length(encoded) - 1))
        if residue_index_offset > 0
            start = 1
            n_chains = length(chains)
            for (i, chain) in enumerate(chains)
                len_chain = length(chain)
                if i < n_chains
                    len_chain += length(chain_linker)
                end
                residx[start:(start + len_chain - 1)] .+= (i - 1) * residue_index_offset
                start += len_chain
            end
        end

        offset = 0
        n_chains = length(chains)
        for (i, chain) in enumerate(chains)
            if i > 1
                append!(chain_index, fill(i - 2, length(chain_linker)))
            end
            append!(chain_index, fill(i - 1, length(chain)))
            offset += length(chain)
            if i < n_chains && length(chain_linker) > 0
                linker_mask[(offset + 1):(offset + length(chain_linker))] .= 0
                offset += length(chain_linker)
            end
        end
    else
        gap = Int(chain_linker)
        start = 0
        for (i, chain) in enumerate(chains)
            len_chain = length(chain)
            append!(residx, collect(start:(start + len_chain - 1)))
            append!(chain_index, fill(i - 1, len_chain))
            start = (start + len_chain - 1) + gap
        end
    end

    return encoded, residx, linker_mask, chain_index
end

function batch_encode_sequences(
    sequences::AbstractVector{<:AbstractString};
    residue_index_offset::Int=512,
    chain_linker::Union{AbstractString,Int}="G"^25,
)
    aatype_list = Vector{Vector{Int}}()
    residx_list = Vector{Vector{Int}}()
    linker_mask_list = Vector{Vector{Float32}}()
    chain_index_list = Vector{Vector{Int}}()

    for seq in sequences
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq;
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        push!(aatype_list, aatype_seq)
        push!(residx_list, residx_seq)
        push!(linker_mask_list, linker_mask_seq)
        push!(chain_index_list, chain_index_seq)
    end

    aatype = collate_dense_tensors(aatype_list, 0)
    mask = collate_dense_tensors([ones(Int, length(x)) for x in aatype_list], 0)
    residx = collate_dense_tensors(residx_list, 0)
    linker_mask = collate_dense_tensors(linker_mask_list, 0f0)
    chain_index = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index
end

# Categorical lDDT

struct CategoricalMixture
    logits
    v_bins
end

function CategoricalMixture(param, bins::Int=50, start::Real=0, stop::Real=1)
    v = range(start, stop; length=bins+1)
    centers = (v[1:end-1] .+ v[2:end]) ./ 2
    v_bins = reshape(centers, ntuple(_ -> 1, ndims(param) - 1)..., length(centers))
    v_bins = to_device(v_bins, param, eltype(param))
    return CategoricalMixture(param, v_bins)
end

function Statistics.mean(cm::CategoricalMixture)
    probs = NNlib.softmax(cm.logits; dims=ndims(cm.logits))
    return sum(probs .* cm.v_bins; dims=ndims(cm.logits))
end

function categorical_lddt(logits; bins::Int=50)
    v = range(0f0, 1f0; length=bins + 1)
    centers = (v[1:end-1] .+ v[2:end]) ./ 2
    shape = ntuple(i -> (i == ndims(logits) ? length(centers) : 1), ndims(logits))
    v_bins = reshape(centers, shape...)
    v_bins = to_device(v_bins, logits, eltype(logits))
    probs = NNlib.softmax(logits; dims=ndims(logits))
    out = sum(probs .* v_bins; dims=ndims(logits))
    return dropdims(out; dims=ndims(logits))
end
