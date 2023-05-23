struct SRLFishTail1D <: ImmersedBoundary
    ρ::Float64 # solid density (kg/m^3)
    nlinks::Int64 # number of links
    linklengths::Vector{Float64} # lengths of each links
    joints::Vector{Int64} # lengths of each links
    x1::Vector{Float64} # [x; y] coords of first link tip
    x_b::Vector{Float64} # [x; y] coords of boundary nodes in initial body frame
    ds::Vector{Float64} # length corresponding to each boundary node
    dz::Vector{Float64} # height corresponding to each boundary node
    nodes::Int64 # number of boundary nodes
    normalize::Bool # are parameters normalized
end

function SRLFishTail1D(ρ::Float64=1.0, x1::AbstractVector=[1.0, 1.0], nominal_ds=0.02)

    linklengths = 1e-3.*[8.68, 9.55, 8.53, 9.14, 9.88, 11.30, 9.28, 9.29, 8.88, 5.69, 19.27]
    jointheights = 1e-3.*[49.00, 48.54, 46.96, 45.17, 41.28, 37.05, 30.51, 23.03, 33.27, 52.06, 59.32, 0.0]
    nlinks = length(linklengths)

    x_b = [0.0]
    ds = [linklengths[1] / ceil(linklengths[1]/nominal_ds)]
    dz = [jointheights[1]]
    joints = ones(nlinks+1)

    for link in 1:nlinks

        l = linklengths[link]

        l_density = Int(ceil(l/nominal_ds))
        x_l = LinRange(x_b[end], x_b[end]+l, l_density+1)
        ds_l = l/l_density .* ones(length(x_l[2:end]))
        dz_l = LinRange(jointheights[link], jointheights[link+1], l_density+1)[2:end]
        joints[link+1] = joints[link] + l_density

        ds = vcat(ds, ds_l)
        dz = vcat(dz, dz_l)
        x_b = vcat(x_b, x_l[2:end])

    end

    y_b = zeros(length(x_b))
    
    # make immersed boundary model
    SRLFishTail1D(ρ, nlinks, linklengths, joints[1:end-1], x1, vcat(x_b, y_b), ds, dz, length(x_b), false)

end

function normalize(model::SRLFishTail1D, ref_L::AbstractFloat)

    nmodel = SRLFishTail1D(model.ρ, model.nlinks, model.linklengths ./ ref_L,
        model.joints, model.x1 ./ ref_L, model.x_b ./ ref_L, model.ds ./ ref_L,
        model.dz ./ ref_L, model.nodes, true)

    return nmodel

end
function normalize(model::SRLFishTail1D, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)
    
    xn = deepcopy(x)
    xn[model.nlinks+1:end] = x[model.nlinks+1:end] ./ (ref_U/ref_L)

    return xn

end

function unnormalize(model::SRLFishTail1D, ref_L::AbstractFloat)

    nmodel = SRLFishTail1D(model.ρ, model.nlinks, model.linklengths .* ref_L,
        model.joints, model.x1 .* ref_L, model.x_b .* ref_L, model.ds .* ref_L,
        model.dz .* ref_L, model.nodes, true)

    return nmodel

end
function unnormalize(model::SRLFishTail1D, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)

    xn = deepcopy(x)
    xn[model.nlinks+1:end] = x[model.nlinks+1:end] .* (ref_U/ref_L)

    return xn

end

function boundary_state(model::SRLFishTail1D, x::AbstractVector)

    xn_b = ones(length(model.x_b)*2)
    boundary_state!(model, xn_b, x)

    return xn_b

end

function boundary_state!(model::SRLFishTail1D, xn_b::AbstractVector, x::AbstractVector)

    joints = model.joints
    nlinks = model.nlinks
    nnodes = model.nodes

    xcn_b = model.x_b[1:joints[1]] .+ model.x1[1]
    ycn_b = model.x_b[nnodes+1:nnodes+joints[1]] .+ model.x1[2]
    ucn_b = zeros(length(xcn_b))
    vcn_b = zeros(length(ycn_b))

    for link in 1:nlinks

        if link == nlinks
            l_nodes = joints[link]+1:nnodes
        else
            l_nodes = joints[link]+1:joints[link+1]
        end
        
        netθ = sum(x[1:link])
        netω = sum(x[nlinks+1:link+nlinks])

        joint_x = model.x_b[joints[link]]
        joint_y = model.x_b[joints[link] .+ nnodes]

        x_l = model.x_b[l_nodes]
        y_l = model.x_b[l_nodes .+ nnodes]
        
        x_b_stacked = Rotations.RotMatrix{2}(netθ)*hcat(x_l .- joint_x, y_l .- joint_y)'
        u_b_stacked = netω .* Rotations.RotMatrix{2}(pi/2)*hcat(
            x_b_stacked[1, :], x_b_stacked[2, :])'

        xcn_b = vcat(xcn_b, x_b_stacked[1, :] .+ xcn_b[end])
        ycn_b = vcat(ycn_b, x_b_stacked[2, :] .+ ycn_b[end])
        
        ucn_b = vcat(ucn_b, u_b_stacked[1, :] .+ ucn_b[end])
        vcn_b = vcat(vcn_b, u_b_stacked[2, :] .+ vcn_b[end])

    end

    xn_b .= vcat(xcn_b, ycn_b, ucn_b, vcn_b)

end

function simulate_predefined(model::SRLFishTail1D, X::AbstractVector)

    X_b = Vector{AbstractVector}(undef, length(X))

    for i in eachindex(X)

        x = X[i]
        xn_b = ones(length(model.x_b)*2)
        boundary_state!(model, xn_b, x)

        X_b[i] = xn_b

    end

    return X_b

end

function plot_boundary(model::SRLFishTail1D, x::AbstractVector;
    color=:black, linewidth=5)

    x_b = boundary_state(model, x)[1:end÷2]
    x = x_b[1:model.nodes]
    y = x_b[model.nodes+1:end]

    fig, ax = lines(x, y, color=color, linewidth=linewidth)
    
    return fig, ax
end

function plot_boundary!(model::SRLFishTail1D, x::AbstractVector;
    color=:black, linewidth=5)

    x_b = boundary_state(model, x)[1:end÷2]
    x = x_b[1:model.nodes]
    y = x_b[model.nodes+1:end]

    lines!(x, y, color=color, linewidth=linewidth)
    
end