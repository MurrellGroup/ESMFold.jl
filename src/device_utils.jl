# Device-agnostic helpers

using ChainRulesCore

like(v, x::AbstractArray, args...) = @ignore_derivatives fill!(similar(x, args...), v)

zeros_like(x::AbstractArray, args...) = like(false, x, args...)
ones_like(x::AbstractArray, args...) = like(true, x, args...)

function to_device(x::AbstractArray, like::AbstractArray, ::Type{T}=eltype(x)) where {T}
    return @ignore_derivatives begin
        y = similar(like, T, size(x))
        y .= T.(x)
        y
    end
end

function to_device(x::Number, like::AbstractArray, ::Type{T}=typeof(x)) where {T}
    return @ignore_derivatives T(x)
end
