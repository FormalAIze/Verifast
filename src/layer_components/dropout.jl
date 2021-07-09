

mutable struct Dropout{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    output_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    rate::Float
    activation::F
end

function forward(layer::Dropout,dict::Dict{String,T}) where {T<:AbstractArray}
    dict[layer.name] = dict[layer.input_name]
    return dict
end
