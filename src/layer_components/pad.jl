

mutable struct ZeroPadding2D{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::NTuple{3,Int}
    output_shape::NTuple{3,Int}
    padding::NTuple{4,Int}
    activation::F
end

function forward(layer::ZeroPadding2D,dict::Dict{String,T}) where {T<:AbstractArray}
    return forward(layer,dict[layer.input_name];dict=dict)
end

function forward(layer::ZeroPadding2D,input::Union{Array{T,3},SubArray{T,3}};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    padding_top,_,padding_left,_ = layer.padding
    height,width,_ = layer.input_shape
    res = zeros(Float,layer.output_shape)
    res[padding_top+1:padding_top+height,padding_left+1:padding_left+width,:] = input
    dict[layer.name]=res
    return dict
end

# 批次处理
function forward(layer::ZeroPadding2D,input::Union{Array{T,4},SubArray{T,4}};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    padding_top,_,padding_left,_ = layer.padding
    height,width,_ = layer.input_shape
    res = zeros(Float,layer.output_shape)
    res[padding_top:padding_top+height,padding_left:padding_left+width,:,:] = input
    dict[layer.name]=res
    return dict
end
