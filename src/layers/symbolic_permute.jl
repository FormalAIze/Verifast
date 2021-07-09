# permute
function forward(layer::Permute,dict::Dict{String,SymbolicMatrix})
    sym = dict[layer.input_name]
    input_data,symbolic_input,symbolic_bias = sym.input,sym.symbolic_weights,sym.symbolic_bias
    new_symbolic_input = permutedims(symbolic_input,layer.dims)
    if symbolic_bias != nothing
        bias_upper = permutedims(symbolic_bias.upper, layer.dims)
        bias_lower = permutedims(symbolic_bias.lower, layer.dims)
        bias = permutedims(symbolic_bias.bias, layer.dims)
        new_bias = BiasInterval(bias,bias_lower,bias_upper)
    end
    dict[layer.name] = SymbolicMatrix(input_data,new_symbolic_input,new_bias,sym.input_shape,sym.wrong_nodes)
    return dict
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Permute{Id})
    dict[layer.output_name] = dict[layer.name]
    return dict
end
