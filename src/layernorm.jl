using Statistics

@concrete struct LayerNormFirst <: Onion.Layer
    w
    b
    eps::Float32
end

@layer LayerNormFirst

function LayerNormFirst(dim::Int; eps=1f-5)
    w = ones(Float32, dim)
    b = zeros(Float32, dim)
    return LayerNormFirst(w, b, Float32(eps))
end

# CUDA fast path: fused cuTile kernel (1 launch, 0 intermediate allocations)
# Falls back to multi-kernel path for non-power-of-2 C (cuTile tile constraint)
function (ln::LayerNormFirst)(x::CuArray)
    C = size(x, 1)
    if ispow2(C)
        return fused_layernorm_first(x, ln.w, ln.b, ln.eps)
    end
    # Non-power-of-2 fallback: multi-kernel path on GPU
    shape = ntuple(_ -> 1, ndims(x) - 1)
    w = reshape(ln.w, length(ln.w), shape...)
    b = reshape(ln.b, length(ln.b), shape...)
    μ = Statistics.mean(x; dims=1)
    diff = x .- μ
    σ2 = Statistics.mean(diff .* diff; dims=1)
    inv_std = 1f0 ./ sqrt.(σ2 .+ ln.eps)
    return @. diff * inv_std * w + b
end

"""
    layernorm_inplace!(ln, x) -> x (modified in-place)

In-place LayerNorm for CUDA. Use when the input is a temporary that can be overwritten.
Falls back to out-of-place + copyto! for non-power-of-2 channel dims.
"""
function layernorm_inplace!(ln::LayerNormFirst, x::CuArray)
    C = size(x, 1)
    if ispow2(C)
        return fused_layernorm_first!(x, ln.w, ln.b, ln.eps)
    end
    result = ln(x)
    copyto!(x, result)
    return x
end

# CPU fallback: multi-kernel path
function (ln::LayerNormFirst)(x::AbstractArray)
    shape = ntuple(_ -> 1, ndims(x) - 1)
    w = reshape(ln.w, length(ln.w), shape...)
    b = reshape(ln.b, length(ln.b), shape...)
    μ = Statistics.mean(x; dims=1)
    diff = x .- μ
    σ2 = Statistics.mean(diff .* diff; dims=1)
    inv_std = 1f0 ./ sqrt.(σ2 .+ ln.eps)
    # Fuse normalize + scale + shift into single broadcast kernel
    return @. diff * inv_std * w + b
end

@concrete struct LinearFirst <: Onion.Layer
    weight
    bias
    use_bias::Bool
end

@layer LinearFirst

function LinearFirst(in_dim::Int, out_dim::Int; bias::Bool=true)
    scale = Float32(1 / sqrt(Float32(in_dim)))
    weight = randn(Float32, out_dim, in_dim) .* scale
    b = bias ? zeros(Float32, out_dim) : zeros(Float32, 0)
    return LinearFirst(weight, b, bias)
end

function (m::LinearFirst)(x::AbstractArray)
    in_dim = size(m.weight, 2)
    out_dim = size(m.weight, 1)
    x2 = reshape(x, in_dim, :)
    y2 = m.weight * x2
    if m.use_bias
        y2 .+= m.bias  # in-place: y2 is already a new array from matmul
    end
    out_shape = (out_dim, size(x)[2:end]...)
    return reshape(y2, out_shape)
end
