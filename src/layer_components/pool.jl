
mutable struct MaxPooling2D{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::NTuple{3,Int}
    output_shape::NTuple{3,Int}
    pool_size::NTuple{2,Int}
    strides::NTuple{2,Int}
    padding::Int
    activation::F
end

mutable struct AveragePooling2D{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::NTuple{3,Int}
    output_shape::NTuple{3,Int}
    pool_size::NTuple{2,Int}
    strides::NTuple{2,Int}
    padding::Int
    activation::F
end

mutable struct  GlobalMaxPooling2D{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::NTuple{3,Int}
    output_shape::NTuple{1,Int}
    activation::F
end

mutable struct  GlobalAveragePooling2D{F} <: Layer where F<:ActivationFunction
    name::String
    op_type::String
    input_name::String
    output_name::String
    input_shape::NTuple{3,Int}
    output_shape::NTuple{1,Int}
    activation::F
end

function forward(layer::Union{MaxPooling2D,AveragePooling2D,GlobalMaxPooling2D,GlobalAveragePooling2D},dict::Dict{String,T}) where {T<:AbstractArray}
    return forward(layer,dict[layer.input_name];dict=dict)
end

function forward(layer::MaxPooling2D,input::AbstractArray{T};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    output = pool2d(input,layer,dict;pool_mode=:max)
    return output
end

function forward(layer::AveragePooling2D,input::AbstractArray{T};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    output = pool2d(input,layer,dict;pool_mode=:avg)
    return output
end

function forward(layer::GlobalMaxPooling2D,input::AbstractArray{T};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    output = pool2d(input,layer,dict;pool_mode=:global_max)
    return output
end

function forward(layer::GlobalAveragePooling2D,input::AbstractArray{T};dict::Dict{String,AbstractArray}=Dict{String,AbstractArray}()) where {T<:AbstractFloat}
    output = pool2d(input,layer,dict;pool_mode=:global_avg)
    return output
end

function pool_to_2d(A::Array{T,3},pool_size::NTuple{2,Int},strides::NTuple{2,Int},padding::Int,output_shape::NTuple{3,Int};row::Bool=false) where {T<:AbstractFloat}
    """Reshape (H,W,C) data into (PH*PW, OH*OW*C).
    Args:
        A           (Array): (H,W,C) data.
        pool_size     (Tuple): (PH,PW) filter.
        ......
    Returns:
        Array: Reshaped (PH*PW, OH*OW*C) data.
    """
    in_height,in_width,in_channels = size(A)
    stride_height,stride_width = strides
    out_height,out_width,_ = output_shape
    kernel_height,kernel_width = pool_size
    padding_fm = A
    center_y,center_x = 1,1
    if padding==1
        padding_need_height = (out_height-1)*stride_height + kernel_height - in_height
        padding_need_height = Int(max(padding_need_height,0))
        padding_top = floor(Int64,padding_need_height/2)
        padding_bottom = padding_need_height-padding_top
        padding_need_width = (out_width-1)*stride_width + kernel_width  - in_width
        padding_need_width = Int(max(padding_need_width,0))
        padding_left = floor(Int64,padding_need_width/2)
        padding_right = padding_need_width - padding_left
        center_y = padding_top+1
        center_x = padding_left+1
        padding_fm = zeros(T,in_height+padding_need_height,in_width+padding_need_width,in_channels)
        padding_fm[padding_top+1:padding_top+in_height,padding_left+1:padding_left+in_width,:]=A
    end
    res = nothing # 初始化空变量
    # 横向滑动
    @inbounds @views for i in 1:out_height, j in 1:out_width
        # 取出当前点为中心的k_h*k_w区域
        if padding==1
            conv_area=padding_fm[(i-1)*stride_height+center_y-padding_top:(i-1)*stride_height+center_y-padding_top+kernel_height-1,(j-1)*stride_width+center_x-padding_left:(j-1)*stride_width+center_x-padding_left+kernel_width-1,:]
        else
            conv_area=padding_fm[(i-1)*stride_height+center_y:(i-1)*stride_height+center_y+kernel_height-1,(j-1)*stride_width+center_x:(j-1)*stride_width+center_x+kernel_width-1,:]
        end
        # 将3d图像块重置成2d，水平拼接矩阵
        if i==1 && j==1
            res = reshape(conv_area,(:,in_channels))
        else
            res = hcat(res,reshape(conv_area,(:,in_channels)))
        end
    end
    return res
end

function pool_to_2d(A::Array{T,4},pool_size::NTuple{2,Int},strides::NTuple{2,Int},padding::Int,output_shape::NTuple{3,Int};row::Bool=false) where {T<:AbstractFloat}
    in_height,in_width,in_channels,batch_size = size(A)
    stride_height,stride_width = strides
    out_height,out_width,_ = output_shape
    kernel_height,kernel_width = pool_size
    padding_fm = A
    center_y,center_x = 1,1
    if padding==1
        padding_need_height = (out_height-1)*stride_height + kernel_height - in_height
        padding_need_height = Int(max(padding_need_height,0))
        padding_top = floor(Int64,padding_need_height/2)
        padding_bottom = padding_need_height-padding_top
        padding_need_width = (out_width-1)*stride_width + kernel_width  - in_width
        padding_need_width = Int(max(padding_need_width,0))
        padding_left = floor(Int64,padding_need_width/2)
        padding_right = padding_need_width - padding_left
        center_y = padding_top+1
        center_x = padding_left+1
        padding_fm = zeros(T,in_height+padding_need_height,in_width+padding_need_width,in_channels,batch_size)
        padding_fm[padding_top+1:padding_top+in_height,padding_left+1:padding_left+in_width,:,:]=A
    end
    res = nothing # 初始化空变量
    col = 0
    # 横向滑动
    @inbounds @views for i in 1:out_height, j in 1:out_width
        # 取出当前点为中心的k_h*k_w区域
        if padding==1
            conv_area=padding_fm[(i-1)*stride_height+center_y-padding_top:(i-1)*stride_height+center_y-padding_top+kernel_height-1,(j-1)*stride_width+center_x-padding_left:(j-1)*stride_width+center_x-padding_left+kernel_width-1,:,:]
        else
            conv_area=padding_fm[(i-1)*stride_height+center_y:(i-1)*stride_height+center_y+kernel_height-1,(j-1)*stride_width+center_x:(j-1)*stride_width+center_x+kernel_width-1,:,:]
        end
        # 将3d图像块重置成2d，水平拼接矩阵
        if i==1 && j==1
            res = reshape(conv_area,(:,in_channels,batch_size))
        else
            res = hcat(res,reshape(conv_area,(:,in_channels,batch_size)))
        end
        col += 1
    end
    res = reshape(res,(:,col*in_channels*batch_size))
    return res
end

# inputs的维度要3
function pool2d(inputs::Array{T,3},layer::Union{MaxPooling2D,AveragePooling2D},dict::Dict{String,AbstractArray};pool_mode::Symbol=:max) where {T<:AbstractFloat}
    dim = ndims(inputs)
    @assert(dim == 3,"Number of dims in input, $dim, does not match dims of CNN data.")
    # @show "pool"
    data = pool_to_2d(inputs,layer.pool_size,layer.strides,layer.padding,layer.output_shape,row=false)
    if pool_mode == :max
        # 获取每列最大项数值
        out = maximum(data,dims=1)
    elseif pool_mode == :avg
        # 获取每列均值
        out = mean(data,dims=1)
    else
        throw(MyException("pool_mode exception"))
    end
    # 转置成3d的矩阵
    out_height,out_width,out_channels = layer.output_shape
    # 先根据通道划分成每行为一个通道的数据
    output = reshape(out,(out_channels,out_height*out_width))
    # 再转置成列
    output = output'
    # 再进行2d->3d的reshape
    output = reshape(output,(out_width,out_height,out_channels))
    # 再把宽高转置回来
    output = permutedims(output, (2,1,3))
    dict[layer.name]=output
    return dict
end

function pool2d(inputs::Array{T,3},layer::Union{GlobalMaxPooling2D,GlobalAveragePooling2D},dict::Dict{String,AbstractArray};pool_mode::Symbol=:max) where {T<:AbstractFloat}
    dim = ndims(inputs)
    @assert(dim == 3,"Number of dims in input, $dim, does not match dims of CNN data.")
    if pool_mode == :global_max
        output = max(inputs,dims=[1,2])
    elseif pool_mode == :global_avg
        output = mean(inputs,dims=[1,2])
    else
        throw(MyException("pool_mode exception"))
    end
    output = reshape(output,layer.output_shape)
    dict[layer.name]=output
    return dict
end

# 批次处理
function pool2d(inputs::Array{T,4},layer::Union{MaxPooling2D,AveragePooling2D},dict::Dict{String,AbstractArray};pool_mode::Symbol=:max) where {T<:AbstractFloat}
    data = pool_to_2d(inputs,layer.pool_size,layer.strides,layer.padding,layer.output_shape,row=false)
    if pool_mode == :max
        # 获取每列最大项数值
        out = maximum(data,dims=1)
    elseif pool_mode == :avg
        # 获取每列均值
        out = mean(data,dims=1)
    else
        throw(MyException("pool_mode exception"))
    end
    # 转置成4d的矩阵
    out_height,out_width,out_channels = layer.output_shape
    batch_size = size(inputs,4)
    # 先根据通道划分成每行为一个通道的数据
    output = reshape(out,(out_channels,out_height*out_width,batch_size))
    # 再转置成列
    output = permutedims(output, (2,1,3))
    # 再进行3d->dd的reshape
    output = reshape(output,(out_width,out_height,out_channels,batch_size))
    # 再把宽高转置回来
    output = permutedims(output, (2,1,3,4))
    dict[layer.name]=output
    return dict
end
