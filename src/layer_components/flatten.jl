

mutable struct Flatten{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::NTuple{3,Int}
    output_shape::NTuple{1,Int}
    activation::F
end

function forward(layer::Flatten,dict::Dict{String,T}) where {T<:AbstractArray}
    return forward(layer,dict[layer.input_name];dict=dict)
end

function forward(layer::Flatten,input::Union{Array{T,3},SubArray{T,3}};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    # 数据的维度是 高度,宽度,通道, tf中拉平的原则是按照纵向的每个通道的数据，先行再列拉平
    # 也就是将通道1的(1,1)与通道2的(1,1)、与通道3的(1,1)，水平放在一起
    # 而julia的reshape是从高维到低维的拉平，也就是说从右向左进行计算
    # 因此需要将 高度,宽度,通道  转化成 通道,宽度,高度,这样就达到了正确的拉平数据
    dict[layer.name] = reshape(permutedims(input,(3,2,1)),:)
    return dict
end

# 批次处理
function forward(layer::Flatten,input::Union{Array{T,4},SubArray{T,4}};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    # 高度,宽度,通道,批次 => 通道,宽度,高度,批次
    batch_size = size(input,4)
    dict[layer.name] = reshape(permutedims(input,(3,2,1,4)),(:,batch_size))
    return dict
end
