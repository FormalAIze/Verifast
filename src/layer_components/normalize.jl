
function normalize_input(x::Array{Float},mean_std::Tuple{Array{Float},Array{Float}})
    mean,std = mean_std
    mean_length = size(mean,1)
    if mean_length == 1
        x .-= mean
        x ./= std
    else
        ndim = ndims(x)
        if ndim == 4
            x[:,:,1,:] .-=  mean[1]
            x[:,:,2,:] .-=  mean[2]
            x[:,:,3,:] .-=  mean[3]
            x[:,:,1,:] ./=  std[1]
            x[:,:,2,:] ./=  std[2]
            x[:,:,3,:] ./=  std[3]
        else
            x[:,:,1] .-=  mean[1]
            x[:,:,2] .-=  mean[2]
            x[:,:,3] .-=  mean[3]
            x[:,:,1] ./=  std[1]
            x[:,:,2] ./=  std[2]
            x[:,:,3] ./=  std[3]
        end
    end
    return x
end
