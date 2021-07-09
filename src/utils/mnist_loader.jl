using MLDatasets
using LinearAlgebra
import Random

"""
手动解压二进制文件包
# using MLDatasets.BinDeps
# file = "C:/Users/万文杰/.julia/datadeps/CIFAR10/cifar-10-binary.tar.gz"
# run(BinDeps.unpack_cmd(file,dirname(file), ".gz", ".tar"));
"""


mutable struct DataLoader
    cur_id::Int
    order::Vector{Int}
end

DataLoader() = DataLoader(1, Random.shuffle(1:60000))

function next_batch(loader::DataLoader, batch_size::Int32)
    x = zeros(Float32, batch_size, 784)
    y = zeros(Float32, batch_size, 10)
    for i in 1:batch_size
        data, label = MLDatasets.MNIST.traindata(loader.order[loader.cur_id])
        x[i, :] = reshape(data, (28*28))
        y[i, Int(label)+1] = 1.0
        loader.cur_id += 1
        if loader.cur_id > 60000
            loader.cur_id = 1
        end
    end
    x, y
end

# using ProgressMeter

# p = Progress(60000,1,"处理第几张图片: ")
# next!(p)

function mnist_data(normalization::Bool=false)
    train_data, train_label = MLDatasets.MNIST.traindata()
    datax = permutedims(train_data, (3,1,2))
    train_x = Array{Float32,2}(reshape(datax, (60000,28*28)))
    test_data, test_label = MLDatasets.MNIST.testdata()
    datax = permutedims(test_data, (3,1,2))
    test_x = Array{Float32,2}(reshape(datax, (10000,28*28)))
    # train_y_index = axes(label)[1]|>collect
    # train_y[train_y_index, label] = 1.0
    train_y = zeros(Float32, 60000, 10)
    test_y = zeros(Float32, 10000, 10)
    for i in 1:60000
        train_y[i, train_label[i]+1] = 1.0
    end
    # map处理
    # gen(x,n) = map(y->y==x ? 1 : 0,1:n)
    # map(x->gen(x,10),label .+ 1)

    for i in 1:10000
        test_y[i, test_label[i]+1] = 1.0
    end
    # if normalization
    #     train_x /= 255
    #     test_x /= 255
    # end
    train_x,train_y,test_x,test_y
end
function load_test_set(N=10000)
    x = zeros(Float32, N, 784)
    y = zeros(Float32, N, 10)
    for i in 1:N
        data, label = MLDatasets.MNIST.testdata(i)
        x[i, :] = reshape(data, (28*28))
        y[i, Int(label)+1] = 1.0
    end
    x,y
end

function load_mnist_test_data_no_one_hot(dataset::Symbol=:MNIST,mode::Symbol=:FC)# normalization::Bool=false,
    if dataset==:MNIST
        test_data, test_label = MLDatasets.MNIST.testdata()
        if mode==:FC
            test_x = Array{Float,2}(reshape(test_data, (:,10000)))
        elseif mode==:CNN
            datax = permutedims(test_data, (2,1,3))
            test_x = Array{Float,4}(reshape(datax, (28,28,1,10000)))
        else
            print("error of data format  has been occurred: no mode was matched !")
        end
    elseif dataset==:FashionMNIST
        test_data, test_label = MLDatasets.FashionMNIST.testdata()
        if mode==:FC
            test_x = Array{Float,2}(reshape(test_data, (:,10000)))
        elseif mode==:CNN
            datax = permutedims(test_data, (2,1,3))
            test_x = Array{Float,4}(reshape(datax, (28,28,1,10000)))
        else
            print("error of data format  has been occurred: no mode was matched !")
        end
    elseif dataset==:CIFAR10
        test_data, test_label = MLDatasets.CIFAR10.testdata()
        test_x = Array{Float,4}(test_data)
        test_x = permutedims(test_x, (2,1,3,4))
        # 显示的是否images库需要的是（3,32,32）
        # colorview(RGB, permutedims(test_data[:,:,:,1], (3,1,2)))
    else
        print("error of data format  has been occurred: no mode was matched !")
    end
    # if normalization
    #     test_x /= 255
    # end
    test_x,test_label
end
