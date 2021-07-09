
include("load_config.jl")

function get_maximum_label(arr::Array{T};dims=1)where {T<:Real}
    ndim = ndims(arr)
    ndim==1 && (return argmax(arr))
    car_idxs = argmax(arr,dims=dims)
    idxs = Array{Int}(undef,length(car_idxs))
    for i in eachindex(car_idxs)
        @inbounds idxs[i] = car_idxs[i][dims]
    end
    return idxs
end

function evaluate_network(network::Network,dataset::Tuple{Array{<:Real},Array{<:Real}})
    test_x,test_y = dataset
    num_dims = ndims(test_x)
    num_samples = size(test_y,1)
    num_correct = 0.0
    @inbounds @views @showprogress 1 "Evaluating the correctness of the network with dataset..." for sample_index in 1:num_samples
        if num_dims == 2
            predicted_label = (forward(network,test_x[:,sample_index]) |> get_maximum_label) -1
        else
            predicted_label = (forward(network,test_x[:,:,:,sample_index]) |> get_maximum_label) -1
        end
        test_y[sample_index] == predicted_label && (num_correct += 1)
    end
    return num_correct / num_samples
end

function evaluate_network_batch(network::Network,dataset::Tuple{Array{<:Real},Array{<:Real}},batch_size::Int=32)
    test_x,test_y = dataset
    num_dims = ndims(test_x)
    num_samples = size(test_y,1)
    num_correct = 0.0
    start_index = 0
    end_index = 0
    num_batch = ceil(Int,num_samples/batch_size)
    @inbounds @views @showprogress 1 "Evaluating the correctness of the network with dataset..." for i in 1:num_batch
        start_index = (i-1)*batch_size+1
        end_index = i*batch_size
        end_index > num_samples && (end_index = num_samples)
        if num_dims == 2
            predict_out = (forward(network,test_x[:,start_index:end_index]) |> get_maximum_label) .- 1
        else
            predict_out = (forward(network,test_x[:,:,:,start_index:end_index]) |> get_maximum_label) .- 1
        end
        for x in zip(test_y[start_index:end_index],predict_out)
            @inbounds x[1]==x[2] && (num_correct += 1)
        end
    end
    return num_correct / num_samples
end

function check_label(network::Network,img::Array{T},target::Int) where {T<:AbstractFloat}
    output = forward(network,img)
    predict = get_maximum_label(output)
    if predict == target
        return true
    end
    return false
end


function find_spurious_label_arr(sym::SymbolicMatrix,target::Int)
    interval = symbol_to_concrete(sym)
    arr,res_flag = Array{Tuple{Int,Float},1}([]),false
    for i in 1:size(interval.upper,1)
        diff = interval.upper[i]-interval.lower[target]
        if i != target && diff >= 0
            push!(arr,(i,diff))
            res_flag = true
        end
    end
    sort!(arr, by = x -> x[2],rev=true)
    new_arr = Array{Int,1}([])
    for a in arr
        push!(new_arr,a[1])
    end
    return new_arr
end


function output_origin_labels(network::Network,
     dataset::Tuple{Array{<:Real},Array{<:Real}},
     save::Bool=true)
    net_name = splitext(basename(network.path))[1]
    test_x, test_y = dataset
    num_dims = ndims(test_x)
    num_samples = size(test_y, 1)
    file_path = string(net_name, "_", Dates.today(), "_origin", ".txt")
    all_predicted_labels = Array{Array{Int,1},1}([])
    for sample_index in 1:100
        if num_dims == 2
            predicted_labels = sortperm(forward(network,test_x[:,sample_index]), rev=true)
        else
            predicted_labels = sortperm(forward(network,test_x[:,:,:,sample_index]), rev=true)
        end
        push!(all_predicted_labels, predicted_labels)
    end
    if save
        open(file_path, "w") do file
            for array in all_predicted_labels
                writedlm(file, [array[2:10]], ',')
            end
        end
    end
end


function output_spurious_labels(network::Network,dataset::Tuple{Array{<:Real},Array{<:Real}},network_type::Symbol;
    range::UnitRange{Int}=1:1,epsilon::Float=Float(1.0/255.0),
    normalize::Tuple{Array{Float,1},Array{Float,1}}=(Array{Float}([0]),Array{Float}([1])),
    save::Bool=true)
    test_x,test_y = dataset
    total_time = @elapsed begin
        # 获取网络文件的名字，与保存文件名字相互拼接
        net_name = splitext(basename(network.path))[1]
        file_path = string(net_name,"_",Dates.today(),"_eps_",string(epsilon)[1:8],".txt")
        totalarr = Array{Array{Int,1},1}([])
        global no_adv = 0
        # global g_img_i = 0
        @showprogress 1 "计算边界 ..." for img_i in range
            # global g_img_i = img_i
            target = test_y[img_i] + 1
            img = network_type == :FC ? test_x[:,img_i] : test_x[:,:,:,img_i]
            img_ub = normalize_input(min.(img .+ epsilon,Float(1.0)),normalize)
            img_lb = normalize_input(max.(img .- epsilon,Float(0.0)),normalize)
            input = Interval(img_lb,img_ub)
            img = normalize_input(img,normalize)
            arr = get_spurious_labels(network,img,input,target)
            push!(totalarr,arr)  
        end
        @show no_adv
        if save
            open(file_path, "w") do file
                for a in totalarr
                    writedlm(file, [a], ',')
                end
            end
        end
    end
    @show total_time
end


function get_spurious_labels(network::Network,img::Array{Float},input::Interval,target::Int)
    # 原网络分类正确下在进行寻找反例
    if check_label(network,img,target)
        sym_output = forward_network(network,input)
        spurious_labels = find_spurious_label_arr(sym_output, target)
        # @show sym_output.symbolic_bias.upper
        # @show sym_output.symbolic_bias.lower
        if size(spurious_labels,1)==0
            global no_adv += 1
            return Array{Int,1}([target])
        else
            # @show g_img_i,spurious_labels,target
            return spurious_labels
        end
    else
        # println("Original Neural Network predicts error, so it can't to find counterexample!")
        return Array{Int,1}([-1])
    end
end