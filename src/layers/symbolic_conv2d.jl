
function fill_copy!(a::Array{T}, x) where T<:Union{Array{Float},Array{Array{Float}}}
    for i in eachindex(a)
        @inbounds a[i] = copy(x)
    end
    return a
end

fill_copy(v, dims::NTuple{N, Integer}) where {N} = fill_copy!(Array{typeof(v),N}(undef, dims), v)

function symbolic_data_to_2d(A::Array{Array{T,1},3},K_shape::NTuple{4,Int},strides::NTuple{2,Int},padding::Int,output_shape::NTuple{3,Int},model_input_shape::NTuple{3,Int};row::Bool=false) where {T<:AbstractFloat}
    in_height,in_width,in_channels = size(A)
    stride_height,stride_width = strides
    out_height,out_width,_ = output_shape
    kernel_nums,kernel_height,kernel_width,kernel_channels = K_shape
    padding_fm = A
    center_y,center_x = 1,1
    padding_top,padding_left = 0,0
    if padding==1
        padding_need_height = Int(max((out_height - 1) * stride_height + kernel_height - in_height,0))
        padding_top = floor(Int,padding_need_height/2)
        padding_need_width = Int(max((out_width - 1) * stride_width + kernel_width  - in_width,0))
        padding_left = floor(Int,padding_need_width/2)
        center_y = padding_top+1
        center_x = padding_left+1
        padding_fm = fill_copy(zeros(T,reduce(*,model_input_shape)),(in_height+padding_need_height,in_width+padding_need_width,in_channels))
        padding_fm[padding_top+1:padding_top+in_height,padding_left+1:padding_left+in_width,:]=A
    end
    rows = kernel_height*kernel_width*kernel_channels
    res = fill_copy(Array{T}(undef,0),(rows,out_height*out_width))
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

function symbolic_data2d_to_reshape(data_2d::Union{Array{T,2},Array{Array{T,1},2}},shape::NTuple{3,Int}; row::Bool=false) where {T<:AbstractFloat}
    H, W, N = shape
    # ==> (N,W,H)
    data = reshape(data_2d,(N,W,H))
    if row == true
        # ==> (H,W,N)
        data = permutedims(data, (3,2,1))
    else
        # ==> (W,H,N)
        data = permutedims(data, (2,3,1))
    end
    return data
end

function forward(layer::Conv2D,input::Interval)
    input_dims = ndims(input.upper)
    # 断言 校验数据
    @assert(input_dims == 3, "Number of dims in input, $input_dims, does not match dims of RGB image.")
    if length(input.upper_3d_to_1d)==0
        input.upper_3d_to_1d = reshape(input.upper,:)
        input.lower_3d_to_1d = reshape(input.lower,:)
    end
    in_height,in_width,in_channels = size(input.upper)
    lb,ub=input.lower,input.upper
    out_height,out_width,out_channels = layer.output_shape
    kernel_height,kernel_width = layer.kernel_size
    symbolic_out_map = fill_copy(zeros(Float,reduce(*,layer.input_shape)),(out_height,out_width,out_channels))
    # symbolic_bias = zeros(Float,reduce(*,(out_height,out_width,out_channels)))
    bias = zeros(Float,(out_height,out_width,out_channels))
    bias_upper = zeros(Float,(out_height,out_width,out_channels))
    bias_lower = zeros(Float,(out_height,out_width,out_channels))
    symbolic_bias = BiasInterval(bias,bias_lower,bias_upper)
    input_vars = zeros(Float,(in_height,in_width,in_channels))
    conv_time = Base.@elapsed begin
        for out_i in 1:layer.out_channels
            for in_i in 1:layer.in_channels
                kernel = layer.weights[out_i,:,:,in_i]
                s_map = input_vars[:,:,in_i]
                symbolic_res = symbolic_conv2d_init(s_map,kernel,layer,in_i,layer.input_shape)
                symbolic_out_map[:,:,out_i] += symbolic_res
            end
            symbolic_bias.bias[:,:,out_i] .+= layer.bias[out_i]
        end
    end
    # @show conv_time
    dict = Dict{String,SymbolicMatrix}()
    dict[layer.name] = SymbolicMatrix(input,symbolic_out_map,symbolic_bias,layer.input_shape)
    return dict
end

function forward(layer::Conv2D,dict::Dict{String,SymbolicMatrix})
    sym = dict[layer.input_name]
    input_data, symbolic_input, sym_bias = sym.input, sym.symbolic_weights, sym.symbolic_bias
    input_height,input_width,input_channels = size(input_data.upper)
    # 符号传播卷积操作的实现
    conv_time = Base.@elapsed begin
        filters = filter_to_2d(layer.weights,row=true)
        data = symbolic_data_to_2d(symbolic_input,size(layer.weights),layer.strides,layer.padding,layer.output_shape,sym.input_shape,row=true)
        # @time output = parallel_mode == GPU ? gpu_matmul(filters,data) : cpu_matmul(filters,data)
        output = cpu_matmul(filters,data)
        symbolic_out_map = symbolic_data2d_to_reshape(output,layer.output_shape,row=true)
        bias_3d_upper,bias_3d_lower,bias_3d = sym_bias.upper,sym_bias.lower,sym_bias.bias
        bias_2d_upper = data_to_2d(bias_3d_upper,size(layer.weights),layer.strides,layer.padding,layer.output_shape,row=true)
        bias_2d_lower = data_to_2d(bias_3d_lower,size(layer.weights),layer.strides,layer.padding,layer.output_shape,row=true)
        bias_2d = data_to_2d(bias_3d,size(layer.weights),layer.strides,layer.padding,layer.output_shape,row=true)
        bias_upper = max.(filters,0) * bias_2d_upper .+ min.(filters,0) * bias_2d_lower
        bias_lower = max.(filters,0) * bias_2d_lower .+ min.(filters,0) * bias_2d_upper
        bias = filters * bias_2d .+ layer.bias
        bias_upper = symbolic_data2d_to_reshape(bias_upper,layer.output_shape,row=true)
        bias_lower = symbolic_data2d_to_reshape(bias_lower,layer.output_shape,row=true)
        bias = symbolic_data2d_to_reshape(bias,layer.output_shape,row=true)
        symbolic_bias = BiasInterval(bias,bias_lower,bias_upper)
    end
    # @show conv_time
    dict[layer.name] = SymbolicMatrix(input_data,symbolic_out_map,symbolic_bias,sym.input_shape,sym.wrong_nodes)
    return dict
end


# 第一次卷积操作，需要将矩阵与卷积核做加法
function symbolic_conv2d_init(A::Array{T},K::Matrix{T},layer::Conv2D,ic::Int,input_shape::NTuple{3,Int}) where {T<:AbstractFloat}
    kernel_height,kernel_width = size(K)
    in_height,in_width = size(A)
    stride_height,stride_width = layer.strides
    out_height,out_width,_ = layer.output_shape
    padding = layer.padding
    center_y,center_x,center_ye,center_xe = 1,1,in_height,in_width
    if padding==1
        padding_need_height = Int(max((out_height - 1) * stride_height + kernel_height - in_height,0))
        padding_top = floor(Int,padding_need_height/2)
        padding_need_width = Int(max((out_width - 1) * stride_width + kernel_width  - in_width,0))
        padding_left = floor(Int,padding_need_width/2)
        center_y = padding_top+1
        center_x = padding_left+1
        center_ye = padding_top+in_height
        center_xe = padding_left+in_width
    end
    res = fill_copy(zeros(T,input_shape),(out_height,out_width))
    res2 = fill_copy(zeros(T,reduce(*,input_shape)),(out_height,out_width))
    for j in 1:out_width, i in 1:out_height
        if padding==1
            s_y = (i-1)*stride_height+center_y-padding_top
            e_y = (i-1)*stride_height+center_y-padding_top+kernel_height-1
            s_x = (j-1)*stride_width+center_x-padding_left
            e_x = (j-1)*stride_width+center_x-padding_left+kernel_width-1
        else
            s_y = (i-1)*stride_height+center_y
            e_y = (i-1)*stride_height+center_y+kernel_height-1
            s_x = (j-1)*stride_width+center_x
            e_x = (j-1)*stride_width+center_x+kernel_width-1
        end
        # 先初始化为滑动区域的坐标点，再收紧有效区域边界，为了方便最后计算权重分量
        sw_y,ew_y,sw_x,ew_x = s_y,e_y,s_x,e_x
        # 左与上有0填充
        if s_y < center_y && s_x < center_x
            sw_y,sw_x = center_y,center_x
        # 右与上有0填充
        elseif s_y < center_y && e_x > center_xe
            sw_y,ew_x = center_y,center_xe
        # 右与下有0填充
        elseif e_y > center_ye && e_x > center_xe
            ew_y,ew_x = center_ye,center_xe
        # 左与下有0填充
        elseif e_y > center_ye && s_x < center_x
            ew_y,sw_x = center_ye,center_x
        # 上
        elseif s_y < center_y
            sw_y = center_y
        # 左
        elseif s_x < center_x
            sw_x = center_x
        # 下
        elseif e_y > center_ye
            ew_y = center_ye
        # 右
        elseif e_x > center_xe
            ew_x = center_xe
        else
            # 在原有的区域内,不需要改动
        end
        diff_sy,diff_ey,diff_sx,diff_ex = sw_y-s_y,ew_y-e_y,sw_x-s_x,ew_x-e_x
        # 切割卷积核
        conv_k = K[1+diff_sy:kernel_height+diff_ey,1+diff_sx:kernel_width+diff_ex]
        # 使用 sw_y,ew_y,sw_x,ew_x 来计算偏移原始输入的位置
        pos_sy,pos_ey,pos_sx,pos_ex = sw_y-center_y+1,ew_y-center_y+1,sw_x-center_x+1,ew_x-center_x+1
        res[i,j][pos_sy:pos_ey,pos_sx:pos_ex,ic] += conv_k
    end
    for j in 1:out_width, i in 1:out_height
        # 这里的拉平，会将矩阵中的每一列进行连接
        res2[i,j] = reshape(res[i,j],:)
    end
    return res2
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Conv2D{Id})
    dict[layer.output_name] = dict[layer.name]
    return dict
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Conv2D{ReLU})
    return multiple_dims_activation(dict,layer)
end

function multiple_dims_activation(dict::Dict{String,SymbolicMatrix},layer::Layer)
    sym = dict[layer.name]
    input_data, symbolic_input, sym_bias = sym.input, sym.symbolic_weights, sym.symbolic_bias
    height,width,channels = size(symbolic_input)
    cmp_time = Base.@elapsed begin
        for c = 1:channels, j = 1:width, i = 1:height
            w = symbolic_input[i,j,c]
            b = sym_bias.bias[i,j,c]
            upper_b = sym_bias.upper[i,j,c]
            lower_b = sym_bias.lower[i,j,c]
            lowlow = b + lower_b + dot(max.(w,0),input_data.lower_3d_to_1d) + dot(min.(w,0),input_data.upper_3d_to_1d)
            upup = b + upper_b + dot(max.(w,0),input_data.upper_3d_to_1d) + dot(min.(w,0),input_data.lower_3d_to_1d)
            if upup <= 0.0
                # 断开依赖
                symbolic_input[i,j,c] *= Float(0.0)
                if layer.use_bias
                    sym_bias.upper[i,j,c] = Float(0.0)
                    sym_bias.lower[i,j,c] = Float(0.0)
                    sym_bias.bias[i,j,c] = Float(0.0)
                end
            elseif lowlow >= 0
                # 维持依赖
            else
                # 近似计算
                k,b = relu_ub_pn(upup,lowlow)
                # 更新表达式 +b
                symbolic_input[i,j,c] = symbolic_input[i,j,c]*k
                if layer.use_bias
                    sym_bias.bias[i,j,c] = sym_bias.bias[i,j,c]*k
                    sym_bias.upper[i,j,c] = sym_bias.upper[i,j,c]*k + b
                    sym_bias.lower[i,j,c] = sym_bias.lower[i,j,c]*k
                end
            end
        end
    end
    # @show cmp_time
    dict[layer.output_name] = SymbolicMatrix(input_data, symbolic_input, sym_bias, sym.input_shape, sym.wrong_nodes)
    return dict
end
