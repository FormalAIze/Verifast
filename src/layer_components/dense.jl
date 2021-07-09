

mutable struct InputLayer{F} <: Layer where F<:ActivationFunction
    input_shape::Array{Int}
    activation::F
    weights::Matrix{Float}
    bias::Vector{Float}
end

mutable struct Dense{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::NTuple{1,Int}
    output_shape::NTuple{1,Int}
    units::Int
    weights_shape::NTuple{2,Int}
    weights::Matrix{Float}
    bias::Vector{Float}
    use_bias::Bool
    activation::F
    cache::Union{Array{Float},Nothing}
end

# forward(layer::Dense, input::Vector{Float}) = layer.weights*input + layer.bias

function forward(layer::Dense,dict::Dict{String,T}) where {T<:AbstractArray}
    return forward(layer,dict[layer.input_name];dict=dict)
end

# 支持多列批次数据处理
function forward(layer::Dense, input::AbstractArray{T};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    if layer.cache == nothing || size(layer.cache) != (size(layer.weights,1),size(input,2))
        output = layer.use_bias ? (layer.weights*input .+ layer.bias) : (layer.weights*input)
        layer.cache = output
    else
        # 复用缓存矩阵
        output = layer.cache
        mul!(output,layer.weights,input)
        if layer.use_bias
            output .+= layer.bias
        end
    end
    dict[layer.name]=output
    return dict
end
# activation(layer::Layer{ReLU}) = max.(x, zero(eltype(x)))

# Dense() = Dense([0],0,Id(),true,Matrix{Float}(undef,0,0),Vector{Float}(undef,0))
# def call(self, inputs):
#     output = K.dot(inputs, self.kernel)
#     if self.use_bias:
#         output = K.bias_add(output, self.bias, data_format='channels_last')
#     if self.activation is not None:
#         output = self.activation(output)
#     return output
