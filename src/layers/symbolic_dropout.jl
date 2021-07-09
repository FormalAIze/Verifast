

function forward(layer::Dropout,dict::Dict{String,SymbolicMatrix})
    dict[layer.name] = dict[layer.input_name]
    return dict
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Dropout{Id}, layer_index::Int, split_nodes::Dict{String,Node})
    dict[layer.output_name] = dict[layer.name]
    return dict
end
