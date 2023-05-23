struct Bar1D <: RigidBody
    m::Float64 # mass of 2D cross-section (ρA)
    J::Float64 # moment of inertia (m=ρA)
    cl::Float64 # characteristic length (i.e. diameter)
    r_b::Vector{Float64} # [x; y] coords of boundary nodes in body frame
    ds::Vector{Float64} # length corresponding to each boundary node
    rhat::Matrix{Float64} # cross-product (hat) map with r_b
    S::Matrix{Int64} # boundary summation matrix
    R::Matrix{Int64} # rearrangement matrix
    nodes::Int64 # number of boundary nodes
    normalize::Bool # are parameters normalized (true = normalized)

    # NOTE: state vector follows following convention:
    #   [x (inertial), y (inertial), θ, v_x (body), v_y (body), ω]

end

function Bar1D(ρ=1.0, l=1.0; w_effective = 0.001, nodes=100)
    
    # determine x and y coordinates of lagrangian points

    rx0_b = Vector(LinRange(-l/2, l/2, nodes))
    ry0_b = zeros(nodes)
    r0_b = vcat(rx0_b, ry0_b)

    # calculate spatial step size between nodes
    ds = l/nodes .* ones(nodes)

    # inertia and geometric properties
    A = l*w_effective
    m = ρ*A
    J = 1/12*m*l^2

    # boundary summation matrix
    S = kron([1 0; 0 1], ones(nodes)')

    # re-arrangement matrix 1 [x1, x2, ..., y1, y2, ...] -> [x1, y1, x2, y2, ...]
    R = hcat(kron(I(nodes), [1, 0]), kron(I(nodes), [0, 1]))
    
    # cross-product (hat) map with r_b
    rhat = Matrix(BlockDiagonal([[-ry0_b[i] rx0_b[i]] for i in 1:nodes]))*R

    # make immersed boundary model
    Bar1D(m, J, l, r0_b, ds, rhat, S, R, nodes, false)

end

function normalize(model::Bar1D, ref_L::AbstractFloat)

    nmodel = Bar1D(model.m / (ref_L^2), model.J / (ref_L^4),
        model.cl / ref_L, model.r_b ./ ref_L, model.ds ./ ref_L,
        model.rhat ./ ref_L, model.S, model.R, model.nodes, true)

    return nmodel

end
function normalize(model::Bar1D, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)
    
    xn = [x[1]/ref_L, x[2]/ref_L, x[3], x[4]/ref_U, x[5]/ref_U, x[6]/(ref_U/ref_L)]

    return xn

end

function unnormalize(model::Bar1D, ref_L::AbstractFloat)

    nmodel = Bar1D(model.m * (ref_L^2), model.J * (ref_L^4),
        model.cl * ref_L, model.r_b .* ref_L, model.ds .* ref_L,
        model.rhat .* ref_L, model.S, model.R, model.nodes, false)

    return nmodel

end
function unnormalize(model::Bar1D, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)

    xn = [x[1]*ref_L, x[2]*ref_L, x[3], x[4]*ref_U, x[5]*ref_U, x[6]*(ref_U/ref_L)]

    return xn

end

function plot_boundary(model::Bar1D, x::AbstractVector;
    color=:black, linewidth=5)

    x_b = boundary_state(model, x)[1:end÷2]
    x = x_b[1:model.nodes]
    y = x_b[model.nodes+1:end]

    boundary_plot = lines(x, y, color=color, linewidth=linewidth)
    
    return boundary_plot
end

function plot_boundary!(scene::Makie.FigureAxisPlot, model::Bar1D, x::AbstractVector; color=:black, linewidth=5)

    x_b = boundary_state(model, x)[1:end÷2]
    x = x_b[1:model.nodes]
    y = x_b[model.nodes+1:end]

    boundary_plot = lines!(scene, x, y, color=color, linewidth=linewidth)
    
    return boundary_plot
end