
function forward(layer::Dense,input::Interval)
    nodes_num = size(input.upper,1)
    expression_time = Base.@elapsed begin
        symbolic_output = copy(layer.weights)
        bias_shape = size(layer.bias)
        bias = BiasInterval(copy(layer.bias),zeros(Float,bias_shape),zeros(Float,bias_shape))
    end
    # @show expression_time
    dict = Dict{String,SymbolicMatrix}()
    dict[layer.name] = SymbolicMatrix(input,symbolic_output,bias,layer.input_shape)
    return dict
end

function forward(layer::Dense,dict::Dict{String,SymbolicMatrix})
    sym = dict[layer.input_name]
    input_data,symbolic_input,sym_bias = sym.input,sym.symbolic_weights,sym.symbolic_bias
    expression_time = Base.@elapsed begin
        if ndims(symbolic_input[1])==1
            symbolic_output = cpu_matmul(layer.weights,symbolic_input)
            # @time symbolic_output = parallel_mode == gpu ? gpu_matmul(layer.weights,symbolic_input) : cpu_matmul(layer.weights,symbolic_input)
            bias = layer.weights * sym_bias.bias + layer.bias
            upper_bias = max.(layer.weights,0) * sym_bias.upper + min.(layer.weights,0) * sym_bias.lower
            lower_bias = max.(layer.weights,0) * sym_bias.lower + min.(layer.weights,0) * sym_bias.upper
        else
            symbolic_output = layer.weights * symbolic_input
            bias = layer.weights * sym_bias.bias + layer.bias
            upper_bias = max.(layer.weights,0) * sym_bias.upper + min.(layer.weights,0) * sym_bias.lower
            lower_bias = max.(layer.weights,0) * sym_bias.lower + min.(layer.weights,0) * sym_bias.upper
        end
    end
    # @show expression_time
    new_bias = BiasInterval(bias,lower_bias,upper_bias)
    dict[layer.name] = SymbolicMatrix(input_data,symbolic_output,new_bias,sym.input_shape,sym.wrong_nodes)
    return dict
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Dense{ReLU})
    return one_dim_activation(dict,layer)
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Dense{Id})
    dict[layer.output_name] = dict[layer.name]
    return dict
end

function activation(dict::Dict{String,SymbolicMatrix}, layer::Dense{Softmax})
    dict[layer.output_name] = dict[layer.name]
    return dict
end

function one_dim_activation(dict::Dict{String,SymbolicMatrix},layer::Layer)
    sym = dict[layer.name]
    interval = symbol_to_concrete(sym)
    input_data,symbolic_input,bias,input_shape = sym.input,sym.symbolic_weights,sym.symbolic_bias,sym.input_shape
    wrong_nodes,mask = sym.wrong_nodes,sym.mask
    dims = ndims(symbolic_input[1])
    n_node, n_input = dims==1 ? (size(symbolic_input,1),0) : size(symbolic_input)
    cmp_time = Base.@elapsed begin
        for i = 1:n_node
            lowlow,upup = interval.lower[i],interval.upper[i]
            if upup <= 0.0
                dims==1 ? symbolic_input[i] *= Float(0.0) : symbolic_input[i,:] *= Float(0.0)
                if layer.use_bias
                    bias.upper[i],bias.lower[i] = Float(0.0),Float(0.0)
                    bias.bias[i] = Float(0.0)
                end
            elseif lowlow >= 0.0
            else
                # 近似计算
                k,b = relu_ub_pn(upup,lowlow)
                # 更新表达式
                dims==1 ? symbolic_input[i] *= k : symbolic_input[i,:] *= k
                if layer.use_bias
                    bias.bias[i] = bias.bias[i]*k
                    bias.upper[i] = bias.upper[i]*k + b
                    bias.lower[i] = bias.lower[i]*k
                end
                # nn += 1
            end
        end
    end
    # @show nn
    # @show cmp_time
    dict[layer.output_name] = SymbolicMatrix(input_data, symbolic_input, bias, sym.input_shape,wrong_nodes)
    return dict
end
