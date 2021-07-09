# flatten
function forward(layer::Flatten,dict::Dict{String,SymbolicMatrix})
    sym = dict[layer.input_name]
    input_data,symbolic_input,symbolic_bias,input_shape = sym.input,sym.symbolic_weights,sym.symbolic_bias,sym.input_shape
    new_symbolic_input = reshape(permutedims(symbolic_input,(3,2,1)),layer.output_shape)
    if symbolic_bias != nothing
        bias_upper = reshape(symbolic_bias.upper,layer.output_shape)
        bias_lower = reshape(symbolic_bias.lower,layer.output_shape)
        bias = reshape(symbolic_bias.bias,layer.output_shape)
        new_bias = BiasInterval(bias,bias_lower,bias_upper)
    end
    dict[layer.name] = SymbolicMatrix(input_data,new_symbolic_input,new_bias,sym.input_shape,sym.wrong_nodes)
    return dict
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Flatten{Id})
    dict[layer.output_name] = dict[layer.name]
    return dict
end
