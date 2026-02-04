# Device-agnostic helpers

function zeros_like(x::AbstractArray, dims::Int...)
    y = similar(x, eltype(x), dims...)
    fill!(y, zero(eltype(x)))
    return y
end

function zeros_like(x::AbstractArray, ::Type{T}, dims::Int...) where {T}
    y = similar(x, T, dims...)
    fill!(y, zero(T))
    return y
end

function ones_like(x::AbstractArray, dims::Int...)
    y = similar(x, eltype(x), dims...)
    fill!(y, one(eltype(x)))
    return y
end

function ones_like(x::AbstractArray, ::Type{T}, dims::Int...) where {T}
    y = similar(x, T, dims...)
    fill!(y, one(T))
    return y
end

function to_device(x::AbstractArray, like::AbstractArray, ::Type{T}=eltype(x)) where {T}
    y = similar(like, T, size(x))
    y .= T.(x)
    return y
end

function to_device(x::Number, like::AbstractArray, ::Type{T}=typeof(x)) where {T}
    return T(x)
end
