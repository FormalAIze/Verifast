
mutable struct Conv2D{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    filter_height::Int
    filter_width::Int
    in_channels::Int
    out_channels::Int  # filters
    filters_shape::NTuple{4,Int}
    input_shape::NTuple{3,Int}
    output_shape::NTuple{3,Int}
    kernel_size::NTuple{2,Int}
    strides::NTuple{2,Int}
    padding::Int
    weights::Array{Float}
    bias::Vector{Float}
    use_bias::Bool
    activation::F
    cache::Union{Array{Float},Nothing}
    cache_map::Union{Array{Float},Nothing}
    cache_res::Union{Array{Float},Nothing}
end

mutable struct SeparableConv2D{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    filter_height::Int
    filter_width::Int
    in_channels::Int
    out_channels::Int
    filters_shape::NTuple{4,Int}
    input_shape::NTuple{3,Int}
    output_shape::NTuple{3,Int}
    kernel_size::NTuple{2,Int}
    strides::NTuple{2,Int}
    padding::Int
    depth_multiplier::Array{Int}
    depthwise_weights::Array{Float}
    pointwise_weights::Array{Float}
    bias::Vector{Float}
    use_bias::Bool
    activation::F
    cache::Union{Array{Float},Nothing}
end

mutable struct DepthwiseConv2D{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    filter_height::Int
    filter_width::Int
    in_channels::Int
    out_channels::Int
    filters_shape::NTuple{4,Int}
    input_shape::NTuple{3,Int}
    output_shape::NTuple{3,Int}
    kernel_size::NTuple{2,Int}
    strides::NTuple{2,Int}
    padding::Int
    depth_multiplier::Array{Int}
    depthwise_weights::Array{Float}
    bias::Vector{Float}
    use_bias::Bool
    activation::F
    cache::Union{Array{Float},Nothing}
end

function forward(layer::Conv2D,dict::Dict{String,T}) where {T<:AbstractArray}
    return forward(layer,dict[layer.input_name];dict=dict)
end

# 卷积层前馈传播，单个图片
function forward(layer::Conv2D,input::AbstractArray{T,3};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    filters = filter_to_2d(layer.weights,row=true)
    data = data_to_2d(input,size(layer.weights),layer.strides,layer.padding,layer.output_shape,row=true)
    if layer.cache == nothing || size(layer.cache) != (size(filters,1),size(data,2))
        output = layer.use_bias ? (filters * data .+ layer.bias) : (filters * data)
        layer.cache = output
    else
        # 复用缓存矩阵
        output = layer.cache
        mul!(output,filters,data)
        if layer.use_bias
            output .+= layer.bias
        end
    end
    out_map = data2d_to_ori(output,layer.output_shape,row=true)
    dict[layer.name]=out_map
    return dict
end

# 批次计算处理
function forward(layer::Conv2D,input::AbstractArray{T,4};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    filters = filter_to_2d(layer.weights,row=true)
    batch_data = data_to_2d(input,size(layer.weights),layer.strides,layer.padding,layer.output_shape,row=true)
    # 如果第一次计算或者前一次批次处理维度不一致,那么创建缓存矩阵以便下次计算使用
    if layer.cache == nothing || size(layer.cache) != (size(filters,1),size(batch_data,2))
        output = filters * batch_data .+ layer.bias
        layer.cache = output
    else
        # 复用缓存矩阵
        output = layer.cache
        mul!(output,filters,batch_data)
        output .+=layer.bias
    end
    out_map = data2d_to_ori(output,(layer.output_shape...,size(input,4)),row=true)
    dict[layer.name]=out_map
    return dict
end

function data_to_2d_by_cache(layer::Conv2D,A::AbstractArray{T,3},K_shape::NTuple{4,Int},strides::NTuple{2,Int},padding::Int,output_shape::NTuple{3,Int};row::Bool=false) where T <: AbstractFloat
    in_height,in_width,in_channels = size(A)
    stride_height,stride_width = strides
    out_height,out_width,_ = output_shape
    cache_res = layer.cache_res
    kernel_nums,kernel_height,kernel_width,kernel_channels = K_shape
    if padding==1
        padding_need_height = (out_height-1)*stride_height + kernel_height - in_height
        padding_need_height = Int(max(padding_need_height,0))
        padding_top = floor(Int,padding_need_height/2)
        padding_bottom = padding_need_height-padding_top
        padding_need_width = (out_width-1)*stride_width + kernel_width  - in_width
        padding_need_width = Int(max(padding_need_width,0))
        padding_left = floor(Int,padding_need_width/2)
        padding_right = padding_need_width - padding_left
        center_y = padding_top+1
        center_x = padding_left+1
        padding_fm = zeros(T,in_height+padding_need_height,in_width+padding_need_width,in_channels)
        padding_fm[padding_top+1:padding_top+in_height,padding_left+1:padding_left+in_width,:]=A
    else
        padding_fm = A
        center_y,center_x = 1,1
    end
    if cache_res==nothing
        cache_res = Array{T}(undef,(kernel_height*kernel_width*kernel_channels,out_height*out_width))
        layer.cache_res = cache_res
    end
    col = 1
    @inbounds @views for i in 1:out_height, j in 1:out_width
        # 取出当前点为中心的k_h*k_w区域
        if padding==1
            conv_area=padding_fm[(i-1)*stride_height+center_y-padding_top:(i-1)*stride_height+center_y-padding_top+kernel_height-1,(j-1)*stride_width+center_x-padding_left:(j-1)*stride_width+center_x-padding_left+kernel_width-1,:]
        else
            conv_area=padding_fm[(i-1)*stride_height+center_y:(i-1)*stride_height+center_y+kernel_height-1,(j-1)*stride_width+center_x:(j-1)*stride_width+center_x+kernel_width-1,:]
        end
        cache_res[:,col] = flatten(conv_area,row=row)
        col += 1
    end
    return cache_res
end
