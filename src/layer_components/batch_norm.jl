

mutable struct BatchNormalization{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    output_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    momentum::Float
    epsilon::Float
    center::Bool
    scale::Bool
    gamma::Array{Float,1}
    beta::Array{Float,1}
    moving_mean::Array{Float,1}
    moving_variance::Array{Float,1}
    activation::F
end

function forward(layer::BatchNormalization,dict::Dict{String,T}) where {T<:AbstractArray}
    return forward(layer,dict[layer.input_name];dict=dict)
end

"""
output = (x - mean) / sqrt(var + epsilon) * gamma + beta
"""

function forward(layer::BatchNormalization,input::Union{Array{T,3},SubArray{T,3}};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    num = length(layer.moving_mean)
    beta = layer.center ? reshape(layer.beta,(1,1,num)) : zeros(Float,(1,1,num))
    gamma = layer.scale ? reshape(layer.gamma,(1,1,num)) : ones(Float,(1,1,num))
    mean = reshape(layer.moving_mean,(1,1,num))
    var = reshape(layer.moving_variance,(1,1,num))
    epsilon = layer.epsilon
    dict[layer.name]=(dict[layer.input_name] .- mean) ./ sqrt.(var .+ epsilon) .* gamma .+ beta
    return dict
end

# 批次处理
function forward(layer::BatchNormalization,input::Union{Array{T,4},SubArray{T,4}};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    num = length(layer.moving_mean)
    mean = reshape(layer.moving_mean,(1,1,num,1))
    variance = reshape(layer.moving_variance,(1,1,num,1))
    dict[layer.name]=(dict[layer.input_name].-mean)./variance
    return dict
end
