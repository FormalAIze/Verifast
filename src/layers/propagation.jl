
include("./symbolic_dense.jl")
include("./symbolic_conv2d.jl")
include("./symbolic_flatten.jl")
include("./symbolic_permute.jl")
include("./symbolic_pad.jl")
include("./symbolic_dropout.jl")

function forward_network(net::Network, input::Interval)::SymbolicMatrix
    for (index, layer) in enumerate(net.layers)
        input = activation(forward(layer, input), layer)
    end
    return input[last(net.layers).output_name]
end

function symbol_to_concrete(sym::SymbolicMatrix)::Interval
    input,weights,bias = sym.input,sym.symbolic_weights,sym.symbolic_bias
    dims = ndims(first(weights))
    n_output, _ = dims==1 ? (size(weights,1),0) : size(weights)
    upper = zeros(Float,n_output)
    lower = zeros(Float,n_output)
    for i in 1:n_output
        w = dims==1 ? weights[i] : weights[i,:]
        b = bias.bias[i]
        upper_b = bias.upper[i]
        lower_b = bias.lower[i]
        if size(w,1) == size(input.lower,1)
            tmplower,tmpupper = input.lower,input.upper
        elseif size(w,1) == size(input.lower_3d_to_1d,1)
            tmplower,tmpupper = input.lower_3d_to_1d,input.upper_3d_to_1d
        end
        upper[i] = b + upper_b + dot(max.(w,0),tmpupper) + dot(min.(w,0),tmplower)
        lower[i] = b + lower_b + dot(max.(w,0),tmplower) + dot(min.(w,0),tmpupper)
    end
    return Interval(lower, upper)
end
