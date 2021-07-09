
abstract type Layer end

Float = Float32

mutable struct Network
    path::String
    layers::Array{Layer}
    maxs::Array{Float}
    mins::Array{Float}
    means::Array{Float}
    ranges::Array{Float}
    Network(a) = new("",a,[],[],[],[])
end
Network() = Network([])
add(m::Network, layer) = push!(m.layers, layer)

abstract type Result end


status(result::Result) = result.status

function validate_status(st::Symbol)
    @assert st ∈ (:holds,:violated, :unknown,:no_solution,:error,:sat, :unsat) "unexpected status code: `:$st`.\nOnly (:holds, :violated, :unknown) are accepted"
    return st
end

mutable struct Res <: Result
    status::Symbol
    epsilon::Union{Int,Float}
    counter_example::Array{Float}
    original_label::Union{Int,String}
    perturbed_label::Union{Int,String}
    predicted_label::Union{Int,String}
    solve_status::String
    objective_value::Float
    Res(s) = new(validate_status(s), 0, Array{Float}(undef,0),"", "", "", "",Float(0.0))
    Res(s,ori,per,pre,st,ov) = new(validate_status(s),0,Array{Float}(undef,0),ori,per,pre,st,ov)
    Res(s,ce,ori,per,pre,st,ov) = new(validate_status(s),0,ce,ori,per,pre,st,ov)
end

mutable struct Interval
    lower::Array{Float}
    upper::Array{Float}
    lower_3d_to_1d::Array{Float}
    upper_3d_to_1d::Array{Float}
    ori_pad::NTuple{4,Int}
    Interval(l, u) = new(l, u, Array{Float}(undef,0), Array{Float}(undef,0),(0,0,0,0))
    Interval(l, u,ll,uu) = new(l, u, ll,uu,(0,0,0,0))
end

mutable struct BiasInterval
    bias::Union{Nothing,Float,Array{Float},Array{Array{Float,1},1}}
    lower::Union{Nothing,Float,Array{Float},Array{Array{Float,1},1}}
    upper::Union{Nothing,Float,Array{Float},Array{Array{Float,1},1}}
end

mutable struct Mask
    names::Array{String,1}
    LG_masks::Array{Array{Int,1},1}
    UG_masks::Array{Array{Int,1},1}
    Mask() = new(Array{String}(undef,0),Array{Array{Int,1},1}(undef,0),Array{Array{Int,1},1}(undef,0))
end

mutable struct Node
    name::String
    status::Int
    weight::Array{Float}
    bias::Float
    grad::Float
    layer_index::Int
    index::Int
    Node(n) = new(n,0,Array{Float,1}(undef,0),Float(0.0),Float(0.0),0,0)
    Node(n,w,b,l) = new(n,0,w,b,Float(0.0),l,0)
    Node(n,s,w,b,g,l,i) = new(n,s,w,b,g,l,i)
end

mutable struct SymbolicMatrix
    input::Interval
    symbolic_weights::Union{Array{Float,2},Array{Array{Float,1},1},Array{Array{Float,3},1},Array{Array{Float,3},2},Array{Array{Float,3},3},Array{Array{Float,1},3}}
    symbolic_bias::BiasInterval
    input_shape::Union{NTuple{1,Int},NTuple{3,Int}}
    wrong_nodes::Array{Node,1}
    mask::Mask
    cache_upper_data::Union{Array{Float,3},Nothing}
    cache_lower_data::Union{Array{Float,3},Nothing}
    SymbolicMatrix(a,b,c,d) = new(a,b,c,d,Array{Node}(undef,0),Mask(),nothing,nothing)
    SymbolicMatrix(a,b,c,d,e) = new(a,b,c,d,e,Mask(),nothing,nothing)
    SymbolicMatrix(a,b,c,d,e,m,f,g) = new(a,b,c,d,e,m,f,g)
end
