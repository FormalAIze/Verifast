

mutable struct Concatenate{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_names::Array{String,1}
    output_name::String
    input_shapes::Union{Array{NTuple{1,Int},1},Array{NTuple{3,Int},1}}
    output_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    activation::F
end

function forward(layer::Concatenate,dict::Dict{String,T}) where {T<:AbstractArray}
    list = []
    for name in layer.input_names
        push!(list,dict[name])
    end
    res = cat(list...;dims=3)
    dict[layer.name] = res
    return dict
end
