

mutable struct Add{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_names::Array{String,1}
    output_name::String
    input_shapes::Union{Array{NTuple{1,Int},1},Array{NTuple{3,Int},1}}
    output_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    activation::F
end

function forward(layer::Add,dict::Dict{String,T}) where {T<:AbstractArray}
    res = zeros(Float,layer.output_shape)
    for name in layer.input_names
        res += dict[name]
    end
    dict[layer.name] = res
    return dict
end


mutable struct Activation{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    output_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    activation::F
end

function forward(layer::Activation,dict::Dict{String,T}) where {T<:AbstractArray}
    dict[layer.name] = dict[layer.input_name]
    return dict
end
