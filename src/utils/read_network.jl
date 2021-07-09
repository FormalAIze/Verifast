using JSON,Random
using BSON
using BSON: @save, @load

function generate_activation(str::String)
    if occursin("relu", str)
        return ReLU()
    elseif occursin("id", str) || occursin("Identity", str) || occursin("linear", str)
        return Id()
    elseif occursin("softmax", str)
        # return Softmax()
        return Id()
    end
    return Id()
end

c_tuple(T,a) = tuple(convert(Array{T,1},a)...)
c_array(T,a) = convert(Array{T,1},a)

function read_network_by_bson(file_path::String)
    data = BSON.load(file_path)[:data]
    model = Network()
    model.path = file_path
    for obj in data
        if occursin("Dense", obj[:op_type])
            weights_shape = c_tuple(Int,obj[:weights_shape])
            layer = Dense(obj[:layer_name],
                          obj[:op_type],
                          obj[:input_name],
                          obj[:output_name],
                          c_tuple(Int,obj[:input_shape]),
                          c_tuple(Int,obj[:output_shape]),
                          Int(obj[:units]),
                          weights_shape,
                          reshape(c_array(Float,obj[:weights]),weights_shape),
                          c_array(Float,obj[:bias]),
                          obj[:use_bias],
                          generate_activation(obj[:activation]),
                          nothing)
        elseif occursin("Conv2D", obj[:op_type])
            filters_shape = c_tuple(Int,obj[:filters_shape])
            weights = c_array(Float,obj[:weights])
            weights = reshape(weights,filters_shape)
            weights = permutedims(weights,(4,1,2,3))
            layer = Conv2D(obj[:layer_name],
                            obj[:op_type],
                            obj[:input_name],
                            obj[:output_name],
                            Int(obj[:filter_height]),
                            Int(obj[:filter_width]),
                            Int(obj[:in_channels]),
                            Int(obj[:out_channels]),
                            filters_shape,
                            c_tuple(Int,obj[:input_shape]),
                            c_tuple(Int,obj[:output_shape]),
                            c_tuple(Int,obj[:kernel_size]),
                            c_tuple(Int,obj[:strides]),
                            Int(obj[:padding]),
                            weights,
                            c_array(Float,obj[:bias]),
                            obj[:use_bias],
                            generate_activation(obj[:activation]),
                            nothing,
                            nothing,
                            nothing)
        elseif occursin("Flatten", obj[:op_type])
            layer = Flatten(obj[:layer_name],
                            obj[:op_type],
                            obj[:input_name],
                            obj[:output_name],
                            c_tuple(Int,obj[:input_shape]),
                            c_tuple(Int,obj[:output_shape]),
                            generate_activation(obj[:activation]))
        elseif occursin("Permute", obj[:op_type])
            layer = Permute(obj[:layer_name],
                            obj[:op_type],
                            obj[:input_name],
                            obj[:output_name],
                            c_tuple(Int,obj[:input_shape]),
                            c_tuple(Int,obj[:output_shape]),
                            c_tuple(Int,obj[:dims]),
                            generate_activation(obj[:activation]))
        elseif occursin("Dropout", obj[:op_type])
            layer = Dropout(obj[:layer_name],
                            obj[:op_type],
                            obj[:input_name],
                            obj[:output_name],
                            c_tuple(Int,obj[:input_shape]),
                            c_tuple(Int,obj[:output_shape]),
                            Float(obj[:rate]),
                            generate_activation(obj[:activation]))
        elseif occursin("ZeroPadding2D", obj[:op_type])
            layer = ZeroPadding2D(obj[:layer_name],
                                obj[:op_type],
                                obj[:input_name],
                                obj[:output_name],
                                c_tuple(Int,obj[:input_shape]),
                                c_tuple(Int,obj[:output_shape]),
                                c_tuple(Int,obj[:padding]),
                                generate_activation(obj[:activation]))
        elseif occursin("Activation", obj[:op_type])
            layer = Activation(obj[:layer_name],
                                obj[:op_type],
                                obj[:input_name],
                                obj[:output_name],
                                c_tuple(Int,obj[:input_shape]),
                                c_tuple(Int,obj[:output_shape]),
                                generate_activation(obj[:activation]))
        else
            @assert(false, string("Can not support layer : ",obj[:op_type]))
        end
        add(model,layer)
    end
    return model
end
