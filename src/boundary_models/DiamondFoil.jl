struct DiamondFoil <: RigidBody
    m::Float64 # mass of 2D cross-section (ρA)
    J::Float64 # moment of inertia (m=ρA)
    cl::Float64 # characteristic length (i.e. diameter)
    r_b::Vector{Float64} # [x; y] coords of boundary nodes in body frame
    ds::Vector{Float64} # length corresponding to each boundary node
    rhat::Matrix{Float64} # cross-product (hat) map with r_b
    S::Matrix{Int64} # boundary summation matrix
    R::Matrix{Int64} # rearrangement matrix
    nodes::Int64 # number of boundary nodes
    leading_ratio::Float64 # ratio of length of leading edge to trailing edge
    normalize::Bool # are parameters normalized (true = normalized)

    # NOTE: state vector follows following convention:
    #   [x (inertial), y (inertial), θ, v_x (body), v_y (body), ω]

end

function DiamondFoil(ρ=1.0, θ=pi/6, c=1.0; leading_ratio=1.0, nodes=100)
    
    nodes_e_vec = round.(Int, nodes.*[leading_ratio / (2 + 2*leading_ratio), 1 / (2 + 2*leading_ratio),
        1 / (2 + 2*leading_ratio), leading_ratio / (2 + 2*leading_ratio)]
    )

    nodes = sum(nodes_e_vec)
    θ2 = asin((1/leading_ratio)*sin(θ))

    # determine x and y coordinates of lagrangian points
    vertices_x = [-c*leading_ratio*cos(θ2), 0, c*cos(θ), 0]
    vertices_y = [0, c*sin(θ), 0, -c*sin(θ)]
 
    rx0_b = zeros(nodes)
    ry0_b = zeros(nodes)

    for i in 0:2

        nodes_e = nodes_e_vec[i+1]
        ind_l = sum(nodes_e_vec[1:i]) + 1
        ind_b = sum(nodes_e_vec[1:i]) + nodes_e

        rx0_b[ind_l:ind_b] = LinRange(vertices_x[i+1], vertices_x[i+2], nodes_e+1)[1:end-1]
        ry0_b[ind_l:ind_b] = LinRange(vertices_y[i+1], vertices_y[i+2], nodes_e+1)[1:end-1]

    end

    nodes_e = nodes_e_vec[4]
    ind_l = sum(nodes_e_vec[1:3]) + 1
    ind_b = sum(nodes_e_vec[1:3]) + nodes_e

    rx0_b[ind_l:ind_b] = LinRange(vertices_x[4], vertices_x[1], nodes_e+1)[1:end-1]
    ry0_b[ind_l:ind_b] = LinRange(vertices_y[4], vertices_y[1], nodes_e+1)[1:end-1]

    # calculate spatial step size between nodes
    ds = (c/nodes_e) .* ones(length(rx0_b))

    # inertia and geometric properties
    A = c^2*sin(θ)cos(θ)
    m = ρ*A
    J = m*((c^4)/6)

    # boundary summation matrix
    S = kron([1 0; 0 1], ones(nodes)')

    # re-arrangement matrix 1 [x1, x2, ..., y1, y2, ...] -> [x1, y1, x2, y2, ...]
    R = hcat(kron(I(nodes), [1, 0]), kron(I(nodes), [0, 1]))
    
    # cross-product (hat) map with r_b
    rhat = Matrix(BlockDiagonal([[-ry0_b[i] rx0_b[i]] for i in 1:nodes]))*R

    # make immersed boundary model
    DiamondFoil(m, J, c, vcat(rx0_b, ry0_b), ds, rhat, S, R, nodes, leading_ratio, false)

end

function normalize(model::DiamondFoil, ref_L::AbstractFloat)

    nmodel = DiamondFoil(model.m / (ref_L^2), model.J / (ref_L^4),
        model.cl / ref_L, model.r_b ./ ref_L, model.ds ./ ref_L,
        model.rhat ./ ref_L, model.S, model.R, model.nodes, model.leading_ratio, true)

    return nmodel

end
function normalize(model::DiamondFoil, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)
    
    xn = [x[1]/ref_L, x[2]/ref_L, x[3], x[4]/ref_U, x[5]/ref_U, x[6]/(ref_U/ref_L)]

    return xn

end

function unnormalize(model::DiamondFoil, ref_L::AbstractFloat)

    nmodel = DiamondFoil(model.m * (ref_L^2), model.J * (ref_L^4),
        model.cl * ref_L, model.r_b .* ref_L, model.ds .* ref_L,
        model.rhat .* ref_L, model.S, model.R, model.nodes, model.leading_ratio, false)

    return nmodel

end
function unnormalize(model::DiamondFoil, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)

    xn = [x[1]*ref_L, x[2]*ref_L, x[3], x[4]*ref_U, x[5]*ref_U, x[6]*(ref_U/ref_L)]

    return xn

end

function plot_boundary(model::DiamondFoil, x::AbstractVector; color=:black)

    x_b = boundary_state(model, x)
    x = x_b[1:model.nodes]
    y = x_b[model.nodes+1:2*model.nodes]

    fig, ax = poly(Point2f[(x[i], y[i]) for i in eachindex(x)], color=color)
    
    return fig, ax
    
end
function plot_boundary!(model::DiamondFoil, x::AbstractVector; color=:black)

    x_b = boundary_state(model, x)
    x = x_b[1:model.nodes]
    y = x_b[model.nodes+1:2*model.nodes]

    poly!(Point2f[(x[i], y[i]) for i in eachindex(x)], color=color)
    
end