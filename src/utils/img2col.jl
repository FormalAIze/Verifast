
function get_ori_argmax_by_views(a::AbstractArray;dims=:)
    ori_idx = parentindices(a)
    ori_row,ori_col = ori_idx[1].start,ori_idx[2].start
    car_ori_idx = CartesianIndex(ori_row-1,ori_col-1)
    car_idxs = argmax(a,dims=dims)
    dims == Colon() && (return car_idxs + car_ori_idx)
    @inbounds for i in eachindex(car_idxs)
        car_idxs[i] += car_ori_idx
    end
    return car_idxs
end


function cpu_matmul(mat1::Array{T,2},mat2::Array{Array{T,1},2}) where {T<:AbstractFloat}
    vars_coef = length(first(mat2))
    mat3 = Array{T}(undef, (vars_coef,size(mat2,1),size(mat2,2)))
    mat4 = fill_copy(Array{T,1}(undef,0), (size(mat1,1),size(mat2,2)))
    mat5 = Array{T}(undef, (vars_coef,size(mat1,1),size(mat2,2)))
    @inbounds for j in 1:size(mat2,2), i in 1:size(mat2,1)
         mat3[:,i,j] = mat2[i,j]
    end
    tmat1 = mat1'
    @inbounds @views for col2 in 1:size(mat2,2)
        mat5[:,:,col2] = mat3[:,:,col2]*tmat1
    end
    @inbounds for col2 in 1:size(mat2,2), col in 1:size(tmat1,2)
        mat4[col,col2] = mat5[:,col,col2]
    end
    mat3,mat5,tmat1 = nothing,nothing,nothing
    return mat4
end

function cpu_matmul(mat1::Array{T,2},mat2::Array{Array{T,1},1}) where {T<:AbstractFloat}
    mat3 = Array{T}(undef, (length(first(mat2)),size(mat2,1)))
    mat4 = fill_copy(Array{T,1}(undef,0), (size(mat1,1),))
    @inbounds for i in 1:size(mat2,1)
         mat3[:,i] = mat2[i]
    end
    tmat3 = mat3'
    s = mat1*tmat3
    s = s'
    @inbounds for col in 1:size(s,2)
        mat4[col] = s[:,col]
    end
    mat3,tmat3,s = nothing,nothing,nothing
    return mat4
end

# function gpu_matmul(mat1::Array{T,2},mat2::Array{Array{T,1},2}) where {T<:AbstractFloat}
#     vars_coef = length(first(mat2))
#     mat3 = Array{T}(undef, (vars_coef,size(mat2,1),size(mat2,2)))
#     mat4 = fill_copy(Array{T,1}(undef,0), (size(mat1,1),size(mat2,2)))
#     mat5 = Array{T}(undef, (vars_coef,size(mat1,1),size(mat2,2)))
#     @inbounds for j in 1:size(mat2,2), i in 1:size(mat2,1)
#          mat3[:,i,j] = mat2[i,j]
#     end
#     @show "transfer_time_gpu"
#     @time begin
#     g_mat3 = CuArray(mat3)
#     end
#     g_mat1 = CuArray(mat1')
#     # carindices = CartesianIndices((size(mat1,1),size(mat2,2)))
#     # @inbounds for idx in eachindex(carindices)
#     #     col,col2 = idx[1],idx[2]
#     #     mat4[col,col2] = Array(CuArrays.@sync mat3[:,:,col2]*tmat1[:,col])
#     # end
#     @show "loop_compute_time_gpu"
#     @time begin
#     @inbounds @views for col2 in 1:size(mat2,2)
#         mat5[:,:,col2] = Array(CuArrays.@sync g_mat3[:,:,col2]*g_mat1)
#     end
#     end
#     @show "transfer_time2"
#     @time begin
#     @inbounds @views for col2 in 1:size(mat5,3), col in 1:size(mat5,2)
#         mat4[col,col2] = mat5[:,col,col2]
#     end
#         # mat4 = reshape(cat(mat5...;dims=2),(:,size(mat5)...))
#         # return Iterators.flatten(a) |> x->reshape(collect(x), :,size(a)...)
#     end
#     # free memory
#     mat3,mat5,g_mat3,g_mat1 = nothing,nothing,nothing,nothing
#     return mat4
# end
#
# function gpu_matmul(mat1::Array{T,2},mat2::Array{Array{T,1},1}) where {T<:AbstractFloat}
#     mat3 = Array{T}(undef, (length(first(mat2)),size(mat2,1)))
#     mat4 = fill_copy(Array{T,1}(undef,0), (size(mat1,1),))
#     @inbounds for i in 1:size(mat2,1)
#          mat3[:,i] = mat2[i]
#     end
#     g_mat3 = CuArray(mat3)
#     g_mat1 = CuArray(mat1')
#     s = CuArrays.@sync g_mat3*g_mat1
#     @inbounds for col in 1:size(s,2)
#         mat4[col] = Array(s[:,col])
#     end
#     # free memory
#     mat3,g_mat3,g_mat1 = nothing,nothing,nothing
#     return mat4
# end


function flatten(x::Union{
    AbstractArray{T,2},
    AbstractArray{T,3},
    AbstractArray{AbstractArray{T,1},3},
    AbstractArray{AbstractArray{T,3},3},
    SubArray{Array{T,1},3},
    SubArray{Array{T,3},3}
    };row::Bool=false) where T <: AbstractFloat
    (row == true) && (x = permutedims(x,reverse(collect(1:ndims(x)),1,2)))
    return reshape(x,:)
end

function flatten_batch_2d(x::Union{
    AbstractArray{T,4},
    AbstractArray{AbstractArray{T,1},4},
    AbstractArray{AbstractArray{T,3},4},
    SubArray{Array{T,1},4},
    SubArray{Array{T,3},4}
    };row::Bool=false) where T <: AbstractFloat
    dims_arr = reverse(collect(1:ndims(x)),1,2)
    (row == true) && (x = permutedims(x,dims_arr))
    return reshape(x,(:,size(x,4)))
end

function filter_to_2d(filters::Array{T,4};row::Bool=false) where T <: AbstractFloat
    """
    kernel  [out_channels,height,width,in_channels]
    Reshape (OC,H,W,IC) filter into (OC, H*W*IC).
    Args:
        filters (array{T,4}).
    Returns:
        Reshaped (OC, H*W*IC) filter.
    """
    OC,H,W,IC = size(filters)
    # 由于julia机制是列优先所以还得将 H W 行列转置后再进行reshape
    (row == true) && (filters = permutedims(filters, (1,3,2,4)))
    return reshape(filters, (OC,H*W*IC))
end

function filter2d_to_orig(filter_2d::Array{T,2}, shape::NTuple{3,Int}; row::Bool=false) where T <: AbstractFloat
    """Reshape (OC, H*W*IC) filter into (OC,H,W,IC).
    Args:
        filter_2d (Array): (K, H*W*IC) filter.
        shape        ((int, int)): (H, w, IC)
    Returns:
        Array: Reshaped (OC,H,W,IC) filter.
    """
    H, W, IC = shape
    filters = reshape(filter_2d,(size(filter_2d,1), H, W, IC))
    (row == true) && (filters = permutedims(filters, (1,3,2,4)))
    return filters
end

function data_to_2d(A::AbstractArray{T,3},K_shape::NTuple{4,Int},strides::NTuple{2,Int},padding::Int,output_shape::NTuple{3,Int};row::Bool=false) where T <: AbstractFloat
    """Reshape (H,W,C) data into (KH*KW*KC, OH*OW).
    Args:
        A           (Array): (H,W,C) data.
        K_shape     (Tuple): (KN,KH,KW,KC) filter.
        ......
    Returns:
        Array: Reshaped (KH*KW*KC, OH*OW) data.
    """
    in_height,in_width,in_channels = size(A)
    stride_height,stride_width = strides
    out_height,out_width,_ = output_shape
    kernel_nums,kernel_height,kernel_width,kernel_channels = K_shape
    padding_fm = A
    center_y,center_x = 1,1
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
    end
    res = Array{T}(undef,(kernel_height*kernel_width*kernel_channels,out_height*out_width))
    col = 1
    @inbounds @views for i in 1:out_height, j in 1:out_width
        if padding==1
            conv_area=padding_fm[(i-1)*stride_height+center_y-padding_top:(i-1)*stride_height+center_y-padding_top+kernel_height-1,(j-1)*stride_width+center_x-padding_left:(j-1)*stride_width+center_x-padding_left+kernel_width-1,:]
        else
            conv_area=padding_fm[(i-1)*stride_height+center_y:(i-1)*stride_height+center_y+kernel_height-1,(j-1)*stride_width+center_x:(j-1)*stride_width+center_x+kernel_width-1,:]
        end
        res[:,col] = flatten(conv_area,row=row)
        col += 1
    end
    return res
end

# 批次处理
function data_to_2d(A::AbstractArray{T,4},K_shape::NTuple{4,Int},strides::NTuple{2,Int},padding::Int,output_shape::NTuple{3,Int};row::Bool=false) where T <: AbstractFloat
    """Reshape (H,W,C,N) data into (KH*KW*KC, OH*OW*N).
    Args:
        A           (Array): (H,W,C,N) data.
        K_shape     (Tuple): (KN,KH,KW,KC) filter.
        ......
    Returns:
        Array: Reshaped (KH*KW*KC, OH*OW*N) data.
    """
    in_height,in_width,in_channels,batch_size = size(A)
    stride_height,stride_width = strides
    out_height,out_width,_ = output_shape
    kernel_nums,kernel_height,kernel_width,kernel_channels = K_shape
    padding_fm = A
    center_y,center_x = 1,1
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
        padding_fm = zeros(T,in_height+padding_need_height,in_width+padding_need_width,in_channels,batch_size)
        padding_fm[padding_top+1:padding_top+in_height,padding_left+1:padding_left+in_width,:,:]=A
    end
    res = Array{T}(undef,(kernel_height*kernel_width*kernel_channels,out_height*out_width,batch_size))
    col = 1
    @inbounds @views for i in 1:out_height, j in 1:out_width
        if padding==1
            conv_area=padding_fm[(i-1)*stride_height+center_y-padding_top:(i-1)*stride_height+center_y-padding_top+kernel_height-1,(j-1)*stride_width+center_x-padding_left:(j-1)*stride_width+center_x-padding_left+kernel_width-1,:,:]
        else
            conv_area=padding_fm[(i-1)*stride_height+center_y:(i-1)*stride_height+center_y+kernel_height-1,(j-1)*stride_width+center_x:(j-1)*stride_width+center_x+kernel_width-1,:,:]
        end
        res[:,col,:] = flatten_batch_2d(conv_area,row=row)
        col += 1
    end
    res = reshape(res,(:,(col-1)*batch_size))
    return res
end

function data2d_to_ori(data::Array{T,2},shape::NTuple{3,Int}; row::Bool=false) where T <: AbstractFloat
    """Reshape (N,H*W) data into (H,W,N) or (W,H,N)."""
    H, W, N = shape
    # ==> (N,W,H)
    data = reshape(data,(N,W,H))
    if row == true
        # ==> (H,W,N)
        data = permutedims(data, (3,2,1))
    else
        # ==> (W,H,N)
        data = permutedims(data, (2,3,1))
    end
    return data
end

# 批次处理数据
function data2d_to_ori(data::Array{T,2},shape::NTuple{4,Int}; row::Bool=false) where T <: AbstractFloat
    """Reshape (C,H*W*N) data into (H,W,C,N) or (W,H,C,N)."""
    H, W, C, N = shape
    data = reshape(data,(C,H*W,N))
    # ==> (H*W,C,N)
    data = permutedims(data, (2,1,3))
    if row == true
        # ==> (H,W,C,N)
        data = reshape(data,(W,H,C,N))
        data = permutedims(data, (2,1,3,4))
    else
        # ==> (W,H,C,N)
        data = reshape(data,(W,H,C,N))
    end
    return data
end
