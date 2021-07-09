
function forward(layer::Activation,dict::Dict{String,SymbolicMatrix})
    dict[layer.name] = dict[layer.input_name]
    return dict
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Activation{Id})
    dict[layer.output_name] = dict[layer.name]
    return dict
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Activation{ReLU})
    data = dict[layer.name]
    if ndims(data.input.upper) == 3
        return multiple_dims_activation(dict,layer)
    else
        return one_dim_activation(dict,layer)
    end
end
