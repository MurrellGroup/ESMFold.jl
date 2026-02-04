using Statistics

@concrete struct LayerNormLast <: Onion.Layer
    w
    b
    eps::Float32
end

@layer LayerNormLast

function LayerNormLast(dim::Int; eps=1f-5)
    w = ones(Float32, dim)
    b = zeros(Float32, dim)
    return LayerNormLast(w, b, Float32(eps))
end

function (ln::LayerNormLast)(x::AbstractArray)
    d = ndims(x)
    μ = Statistics.mean(x; dims=d)
    σ2 = Statistics.mean((x .- μ) .^ 2; dims=d)
    x̂ = (x .- μ) ./ sqrt.(σ2 .+ ln.eps)
    shape = ntuple(_ -> 1, d - 1)
    w = reshape(ln.w, shape..., length(ln.w))
    b = reshape(ln.b, shape..., length(ln.b))
    return x̂ .* w .+ b
end

@concrete struct LinearLast <: Onion.Layer
    weight
    bias
    use_bias::Bool
end

@layer LinearLast

function LinearLast(in_dim::Int, out_dim::Int; bias::Bool=true)
    scale = Float32(1 / sqrt(Float32(in_dim)))
    weight = randn(Float32, out_dim, in_dim) .* scale
    b = bias ? zeros(Float32, out_dim) : zeros(Float32, 0)
    return LinearLast(weight, b, bias)
end

function (m::LinearLast)(x::AbstractArray)
    in_dim = size(m.weight, 2)
    out_dim = size(m.weight, 1)
    x2 = reshape(x, :, in_dim)
    y2 = x2 * m.weight'
    if m.use_bias
        y2 .+= m.bias'
    end
    out_shape = (size(x)[1:end-1]..., out_dim)
    return reshape(y2, out_shape)
end
