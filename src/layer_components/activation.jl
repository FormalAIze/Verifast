abstract type ActivationFunction end

struct ReLU <: ActivationFunction end

struct Max <: ActivationFunction end

struct Id <: ActivationFunction end

struct Softmax <: ActivationFunction end

struct GeneralAct <: ActivationFunction
    f::Function
end

function (f::ReLU)(dict::Dict,layer::Layer)
    x = dict[layer.name]
    x = max.(x, zero(eltype(x)))
    dict[layer.output_name] = x
    return dict
end

function (f::Id)(dict::Dict,layer::Layer)
    dict[layer.output_name] = dict[layer.name]
    return dict
end

function (f::Softmax)(dict::Dict,layer::Layer)
    x = dict[layer.name]
    x = exp.(x)
    x ./= sum(x)
    dict[layer.output_name] = x
    return dict
end
(G::GeneralAct)(dict::Dict,layer::Layer) = G.f(dict,layer)


####################################          relu近似             #############################################
# 上界中
function relu_ub_pn(u::Float,l::Float)::Tuple{Float,Float}
    k = u/(u-l)
    b = -l*k
    return k,b
end
# 下界中
# relu_lb_pn(u::Float,l::Float)::Tuple{Float,Float} = abs(u)>abs(l) ? 1.0 : 0.0, 0.0
relu_lb_pn(u::Float,l::Float)::Tuple{Float,Float} = u/(u-l), 0.0
# 上界右
relu_ub_p(u::Float,l::Float)::Tuple{Float,Float} = 1.0, 0.0
# 下界右
relu_lb_p(u::Float,l::Float)::Tuple{Float,Float} = 1.0, 0.0
# 自适应上界中
function relu_lb_pn_adpative(u::Float,l::Float)::Tuple{Float,Float}
    if abs(u)>=abs(l)
        return relu_ub_p(u,l)
    else
        return relu_lb_pn(u,l)
    end
end
function relu_ub_pn_softplus(u::Float,l::Float)::Tuple{Float,Float}
    mid = (u+l)/2.0
    if u<=1
        ee = 1.3132616875182228
        return max(min(softplus(mid)/abs(mid),1),0.0)/ee,0
    end
    return max(min(softplus(mid)/abs(mid),1),0.0),0
end
####################################          sigmod tanh arctan 近似             #############################################
# tanh导数
function tanh_d(x::Float)::Float
    t = tanh(x)
    return 1-t*t
end
# arctan导数
atan_d(x::Float)::Float = 1.0/(1+x*x)
# sigmod与sigmod导数   expm1(x)==exp(-x)
sigmod(x::Float)::Float = 1.0/(1.0+expm1(x))
sigmod_d(x::Float)::Float = sigmod(x)*(1.0+sigmod(x))
softplus(x::Float)::Float = log(1+exp(x))
softplus_d(x::Float)::Float = sigmod(x)
# 通用上界左
function general_ub_n(u::Float,l::AbstractFloat,func,dfunc)::Tuple{Float,Float}
    k = (func(u)-func(l))/(u-l)
    b = func(l) - l*k
    return k,b
end
# 通用下界左
function general_lb_n(u::Float,l::Float,func,dfunc)::Tuple{Float,Float}
    d = (u+l)/2.0
    k = dfunc(d)
    b = func(d)-d*k
    return k,b
end
# 通用上界右
function general_ub_p(u::Float,l::Float,func,dfunc)::Tuple{Float,Float}
    d = (u+l)/2.0
    k = dfunc(d)
    b = func(d)-d*k
    return k,b
end
# 通用下界右
function general_lb_p(u::Float,l::Float,func,dfunc)::Tuple{Float,Float}
    k = (func(u)-func(l))/(u-l)
    b = func(l) - l * k
    return k, b
end
# 通用上界中
function general_ub_pn(u::Float,l::Float,func,dfunc)::Tuple{Float,Float}
    d_UB = find_d_UB(u,l,func,dfunc)
    k = (func(d_UB)-func(l))/(d_UB-l)
    b  = func(l) - (l - 0.01) * k
    return k, b
end
# 通用下界中
function general_lb_pn(u::Float,l::Float,func,dfunc)::Tuple{Float,Float}
    d_LB = find_d_LB(u,l,func,dfunc)
    k = (func(d_LB)-func(u))/(d_LB-u)
    b = func(u) - (u + 0.01) * k
    return k, b
end
# 寻找中间部分上界的d
function find_d_UB(u::Float,l::Float, func, dfunc)::Float
    max_iter = 10
    d = u/2.0
    ub = u; lb = 0.0
    for _ = 1:max_iter
        t = (func(d)-func(l))/(d-l) - dfunc(d)
        if t > 0 && abs(t) < 0.01          # 增加t>0条件约束会增大开销去搜索绝对误差< 0.01的右边的d点，其实可以求左边一点的d点(-0.01<=diff<=0)
            break
        end
        if t > 0                               # 两点坐标计算的斜率比d点斜率陡峭，也就是需要往左边搜索d
            ub = d
            d = (d+lb)/2.0
        else                                   # 往右边搜索d
            lb = d
            d = (d+ub)/2.0
        end
    end
    return d
end

# 寻找中间部分下界的d
function find_d_LB(u::Float,l::Float,func,dfunc)::Float
    max_iter = 10
    d = l/2.0
    ub = 0.0; lb = l
    for _ = 1:max_iter
        t = (func(u)-func(d))/(u-d) - dfunc(d)
        if t > 0.0 && abs(t) < 0.01
            break
        end
        if t > 0.0
            lb = d
            d = (d+ub)/2.0
        else
            ub = d
            d = (d+lb)/2.0
        end
    end
    return d
end
