struct SRLFishTail2D <: ImmersedBoundary
    ρ::Float64 # solid density (kg/m^3)
    nlinks::Int64 # number of links
    linklengths::Vector{Float64} # lengths of each links
    joints::Vector{Int64} # lengths of each links
    x1::Vector{Float64} # [x; y] coords of first link tip
    x_b::Vector{Float64} # [x; y] coords of boundary nodes in initial body frame
    ds::Vector{Float64} # length corresponding to each boundary node
    dz::Vector{Float64} # height corresponding to each boundary node
    point_order::Vector{Int64} # indices of points in counter clockwise order
    nodes::Int64 # number of boundary nodes
    normalize::Bool # are parameters normalized
end

function SRLFishTail2D(ρ::Float64=1.0, x1::AbstractVector=[1.0, 1.0], nominal_ds=0.02)

    linklengths = 1e-3.*[8.68, 9.55, 8.53, 9.14, 9.88, 11.30, 9.28, 9.29, 8.88, 5.69, 0.002, 19.27]
    jointheights = 1e-3.*[49.00, 48.54, 46.96, 45.17, 41.28, 37.05, 30.51, 23.03, 33.27, 52.06, 59.32, 59.32, 0.0]
    jointwidths = 1e-3.*[40.22, 39.25, 35.50, 32.72, 29.25, 26.65, 23.62, 19.83, 15.12, 14.55, 14.17, 0.55, 0.55]
    nlinks = length(linklengths)

    # joints = Int.(length(x_b).*ones(nlinks+1))

    l_density = Int(ceil(jointwidths[1]/nominal_ds))
    ds = (jointwidths[1] / l_density) .* ones(l_density)
    dz = jointheights[1] .* ones(l_density)
    x_b = zeros(l_density)
    y_b = Vector(LinRange(jointwidths[1]/2, -jointwidths[1]/2, l_density))

    joints = Int.(length(x_b).*ones(nlinks+1))

    for link in 1:nlinks

        l = linklengths[link]

        if jointwidths[link] < 0.75*nominal_ds && jointwidths[link+1] < 0.75*nominal_ds
            
            l_density = Int(ceil(l/nominal_ds))

            x_l = LinRange(x_b[end], x_b[end]+l, l_density+1)[2:end]
            y_l = zeros(length(x_l))
            ds_l = (l/l_density) .* ones(length(x_l))
            dz_l = LinRange(jointheights[link], jointheights[link+1], l_density+1)[2:end]

        elseif jointwidths[link+1] < 0.75*nominal_ds

            jointwidths[link+1] = 0.0

            s = sqrt(l^2 + (jointwidths[link+1]/2-jointwidths[link]/2)^2)
            l_density = Int(ceil(s/nominal_ds))

            x_l = LinRange(x_b[end], x_b[end]+l, l_density+1)[2:end]
            x_l = vcat(x_l[1:end-1], x_l)

            y_l = LinRange(jointwidths[link]/2, jointwidths[link+1]/2, l_density+1)[2:end]
            y_l = vcat(y_l[1:end-1], -y_l[1:end-1], [0.0])

            ds_l = (s/l_density) .* ones(length(x_l))

            dz_l = LinRange(jointheights[link], jointheights[link+1], l_density+1)[2:end]
            dz_l = vcat(dz_l[1:end-1], dz_l)

        else

            s = sqrt(l^2 + (jointwidths[link+1]/2-jointwidths[link]/2)^2)
            l_density = Int(ceil(s/nominal_ds))

            x_l = LinRange(x_b[end], x_b[end]+l, l_density+1)[2:end]
            x_l = vcat(x_l, x_l)

            y_l = LinRange(jointwidths[link]/2, jointwidths[link+1]/2, l_density+1)[2:end]
            y_l = vcat(y_l, -y_l)

            ds_l = (s/l_density) .* ones(length(x_l))

            dz_l = LinRange(jointheights[link], jointheights[link+1], l_density+1)[2:end]
            dz_l = vcat(dz_l, dz_l)

        end

        if link == nlinks

            l_density = Int(ceil(jointwidths[end]/nominal_ds))

            if jointwidths[link+1] < 0.75*nominal_ds
                jointwidths[link+1] = 0.0
            end

            if l_density != 1 && jointwidths[end] != 0
                x_l = vcat(x_l, x_l[end].*ones(l_density-1))

                y_end = LinRange(jointwidths[end]/2, -jointwidths[end]/2, l_density+1)
                y_l = vcat(y_l, y_end[2:end-1])

                ds_l = vcat(ds_l, (jointwidths[end] / l_density) .* ones(l_density-1))
                dz_l = vcat(dz_l, jointheights[end] .* ones(l_density-1))
            end

        end
        
        joints[link+1] = joints[link] + length(x_l)

        ds = vcat(ds, ds_l)
        dz = vcat(dz, dz_l)
        x_b = vcat(x_b, x_l)
        y_b = vcat(y_b, y_l)

    end

    p = sort_points(x_b[1:joints[end-1]], y_b[1:joints[end-1]])
    
    # make immersed boundary model
    SRLFishTail2D(ρ, nlinks, linklengths, joints[1:end-1], x1, vcat(x_b, y_b), ds, dz, p, length(x_b), false)

end

function normalize(model::SRLFishTail2D, ref_L::AbstractFloat)

    nmodel = SRLFishTail2D(model.ρ, model.nlinks, model.linklengths ./ ref_L,
        model.joints, model.x1 ./ ref_L, model.x_b ./ ref_L, model.ds ./ ref_L,
        model.dz ./ ref_L, model.point_order, model.nodes, true)

    return nmodel

end
function normalize(model::SRLFishTail2D, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)
    
    xn = deepcopy(x)
    xn[model.nlinks+1:end] = x[model.nlinks+1:end] ./ (ref_U/ref_L)

    return xn

end

function unnormalize(model::SRLFishTail2D, ref_L::AbstractFloat)

    nmodel = SRLFishTail2D(model.ρ, model.nlinks, model.linklengths .* ref_L,
        model.joints, model.x1 .* ref_L, model.x_b .* ref_L, model.ds .* ref_L,
        model.dz .* ref_L, model.point_order, model.nodes, false)

    return nmodel

end
function unnormalize(model::SRLFishTail2D, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)

    xn = deepcopy(x)
    xn[model.nlinks+1:end] = x[model.nlinks+1:end] .* (ref_U/ref_L)

    return xn

end

function boundary_state(model::SRLFishTail2D, x::AbstractVector)

    xn_b = ones(length(model.x_b)*2)
    boundary_state!(model, xn_b, x)

    return xn_b

end

function boundary_state!(model::SRLFishTail2D, xn_b::AbstractVector, x::AbstractVector)

    joints = model.joints
    nlinks = model.nlinks
    nnodes = model.nodes

    x0_b = model.x_b[1:model.nodes]
    y0_b = model.x_b[model.nodes+1:end]

    # xcn_b = x0_b[1:model.joints[1]] .+ model.x1[1]
    # ycn_b = y0_b[1:model.joints[1]] .+ model.x1[2]

    xcn_b = x0_b[findall(==(0), x0_b)] .+ model.x1[1]
    ycn_b = y0_b[1:length(xcn_b)] .+ model.x1[2]

    x_centerline_joint = xcn_b[end]
    y_centerline_joint = model.x1[2]
    u_centerline_joint = 0.0
    v_centerline_joint = 0.0
    
    ucn_b = zeros(length(xcn_b))
    vcn_b = zeros(length(xcn_b))

    for link in 1:nlinks

        if link == nlinks
            l_nodes = joints[link]+1:nnodes
        else
            l_nodes = joints[link]+1:joints[link+1]
        end
        
        netθ = sum(x[1:link])
        netω = sum(x[nlinks+1:link+nlinks])

        joint_x = x0_b[joints[link]]
        joint_y = 0.0
        
        x_l = x0_b[l_nodes]
        y_l = y0_b[l_nodes]
        
        x_b_stacked = Rotations.RotMatrix{2}(netθ)*hcat(x_l .- joint_x, y_l .- joint_y)'
        u_b_stacked = netω .* Rotations.RotMatrix{2}(pi/2)*hcat(
            x_b_stacked[1, :], x_b_stacked[2, :])'

        xcn_b = vcat(xcn_b, x_b_stacked[1, :] .+ x_centerline_joint)
        ycn_b = vcat(ycn_b, x_b_stacked[2, :] .+ y_centerline_joint)
        
        ucn_b = vcat(ucn_b, u_b_stacked[1, :] .+ u_centerline_joint)
        vcn_b = vcat(vcn_b, u_b_stacked[2, :] .+ v_centerline_joint)

        # determine next joint's centerline position and velocity

        centerline_x = Rotations.RotMatrix{2}(netθ)*[model.linklengths[link], 0.0]
        centerline_u = netω .* Rotations.RotMatrix{2}(pi/2)*centerline_x

        x_centerline_joint += centerline_x[1]
        y_centerline_joint += centerline_x[2]
        u_centerline_joint += centerline_u[1]
        v_centerline_joint += centerline_u[2]

    end

    xn_b .= vcat(xcn_b, ycn_b, ucn_b, vcn_b)

end

function simulate_predefined(model::SRLFishTail2D, X::AbstractVector)

    X_b = Vector{AbstractVector}(undef, length(X))

    for i in eachindex(X)

        x = X[i]
        xn_b = ones(length(model.x_b)*2)
        boundary_state!(model, xn_b, x)

        X_b[i] = xn_b

    end

    return X_b

end

function plot_boundary(model::SRLFishTail2D, x::AbstractVector;
    color=:black, linewidth=5)

    x_b = boundary_state(model, x)[1:end÷2]
    x = x_b[1:model.joints[end]]
    y = x_b[model.nodes+1:model.nodes+model.joints[end]]

    p = model.point_order

    sorted_x = x[p]
    sorted_y = y[p]

    x_fin = x_b[model.joints[end]:model.nodes]
    y_fin = x_b[model.nodes+model.joints[end]:end]

    fig, ax = poly(Point2f[(sorted_x[i], sorted_y[i]) for i in eachindex(sorted_x)],
        color=color, grid=false)
    lines!(ax, x_fin, y_fin, color=color, linewidth=linewidth, grid=false)
    
    return fig, ax
end

function plot_boundary!(model::SRLFishTail2D, x::AbstractVector;
    color=:black, linewidth=5)

    x_b = boundary_state(model, x)[1:end÷2]
    x = x_b[1:model.joints[end]]
    y = x_b[model.nodes+1:model.nodes+model.joints[end]]

    p = model.point_order

    sorted_x = x[p]
    sorted_y = y[p]

    x_fin = x_b[model.joints[end]:model.nodes]
    y_fin = x_b[model.nodes+model.joints[end]:end]

    boundary_plot_1 = poly!(Point2f[(sorted_x[i], sorted_y[i]) for i in eachindex(sorted_x)],
        color=color, grid=false)
    boundary_plot_2 = lines!(x_fin, y_fin, color=color, linewidth=linewidth, grid=false)

    return [boundary_plot_1, boundary_plot_2]
    
end