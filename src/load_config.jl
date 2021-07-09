# 加载使用到的库
using LinearAlgebra,Printf,ProgressMeter,Dates,DelimitedFiles
# CPU并行库
using Base.Threads
# GPU并行库
# using CuArrays,CUDAnative,CUDAdrv,GPUArrays
# using CuArrays: CuArray

# 启用/禁用 CPU、GPU并行标志
@enum ParallelMode CPU=1 GPU=2 turn_off=3 auto=4

# include 顺序不能随意调换，因为有先后类型的依赖关系
include("./layers/data_structure.jl")       # 数据结构类
include("./layer_components/activation.jl")              # 激活函数集合类
include("./utils/img2col.jl")
include("./layer_components/layer.jl")                   # 网络层集合类
include("./utils/read_network.jl")                       # 读取网络工具类
include("./layers/propagation.jl")                       # 传播
include("./utils/mnist_loader.jl")                       # 数据集
# include("./cpu_&_gpu_parallel.jl")
