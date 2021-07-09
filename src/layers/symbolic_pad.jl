

function forward(layer::ZeroPadding2D,input::Interval)
    padding_top,_,padding_left,_ = layer.padding
    input.ori_pad = layer.padding
    height,width,_ = layer.input_shape
    # 如果第一层是pad层需要改变图片层大小，需要同时修改Input_shape
    res = zeros(Float,layer.output_shape)
    res[padding_top+1:padding_top+height,padding_left+1:padding_left+width,:] = input.upper
    input.upper = res
    input.upper_3d_to_1d = reshape(input.upper,:)
    res = zeros(Float,layer.output_shape)
    res[padding_top+1:padding_top+height,padding_left+1:padding_left+width,:] = input.lower
    input.lower = res
    input.lower_3d_to_1d = reshape(input.lower,:)
    return input
end

function forward(layer::ZeroPadding2D,dict::Dict{String,SymbolicMatrix})
    input = dict[layer.input_name]
    input_data,symbolic_input,bias,upper_data, lower_data = input.input,input.symbolic_weights,input.symbolic_bias, input.cache_upper_data, input.cache_lower_data
    padding_top,_,padding_left,_ = layer.padding
    height,width,_ = layer.input_shape
    symbolic_output = fill(zeros(Float,reduce(*,input_shape)),layer.output_shape)
    symbolic_output[padding_top+1:padding_top+height,padding_left+1:padding_left+width,:] = symbolic_input
    dict[layer.name] = SymbolicMatrix(input_data,symbolic_output,bias,upper_data, lower_data)
    return dict
end

function activation(input::Interval, layer::ZeroPadding2D{Id})
    return input
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::ZeroPadding2D{Id})
    dict[layer.output_name] = dict[layer.name]
    return dict
end
