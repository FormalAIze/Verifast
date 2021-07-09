

mutable struct Permute{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::NTuple{3,Int}
    output_shape::NTuple{3,Int}
    dims::NTuple{3,Int}
    activation::F
end

function forward(layer::Permute,dict::Dict{String,T}) where {T<:AbstractArray}
    return forward(layer,dict[layer.input_name];dict=dict)
end

function forward(layer::Permute,input::Union{Array{T,3},SubArray{T,3}};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    dict[layer.name] = permutedims(input,layer.dims)
    return dict
end

# # 批次处理
function forward(layer::Permute,input::Union{Array{T,4},SubArray{T,4}};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}

    dict[layer.name] = permutedims(input,(layer.dims...,4))
    return dict
end
