
# 加载网络层组建
include("./dense.jl")
include("./conv2d.jl")
include("./flatten.jl")
include("./permute.jl")
include("./dropout.jl")
include("./batch_norm.jl")
include("./concatenate.jl")
include("./add.jl")
include("./pad.jl")
include("./pool.jl")
include("./normalize.jl")

# 网络传播计算
function forward(m::Network, x::Union{Array{T},SubArray{T}};output_map::Bool=false) where {T<:AbstractFloat}
    for layer in m.layers
        # @show layer.name
        x = layer.activation(forward(layer, x), layer)
    end
    output_map && return x
    # return typeof(x) == Dict{String,AbstractArray} ? x[last(m.layers).output_name] : x
    return x[last(m.layers).output_name]
end
