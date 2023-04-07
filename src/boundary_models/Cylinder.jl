struct Cylinder <: RigidBody
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

function Cylinder(ρ=1.0, D=0.5; nodes=100)
    
    # determine x and y coordinates of lagrangian points
    θ_b = Vector(LinRange(0, 2*pi, nodes+1))
    rx0_b = D/2 .* cos.(θ_b[1:end-1])
    ry0_b = D/2 .* sin.(θ_b[1:end-1])

    # calculate spatial step size between nodes
    ds = (D*pi/nodes) .* ones(length(rx0_b))

    # inertia and geometric properties
    A = pi*(D/2)^2
    m = ρ*A
    J = 0.5*m*((D/2)^2)

    # boundary summation matrix
    S = kron([1 0; 0 1], ones(nodes)')

    # re-arrangement matrix 1 [x1, x2, ..., y1, y2, ...] -> [x1, y1, x2, y2, ...]
    R = hcat(kron(I(nodes), [1, 0]), kron(I(nodes), [0, 1]))
    
    # cross-product (hat) map with r_b
    rhat = Matrix(BlockDiagonal([[-ry0_b[i] rx0_b[i]] for i in 1:nodes]))*R

    # make immersed boundary model
    Cylinder(m, J, D, vcat(rx0_b, ry0_b), ds, rhat, S, R, nodes, false)

end

function normalize(model::Cylinder, ref_L::AbstractFloat)

    nmodel = Cylinder(model.m / (ref_L^2), model.J / (ref_L^4),
        model.cl / ref_L, model.r_b ./ ref_L, model.ds ./ ref_L,
        model.rhat ./ ref_L, model.S, model.R, model.nodes, true)

    return nmodel

end
function normalize(model::Cylinder, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)
    
    xn = [x[1]/ref_L, x[2]/ref_L, x[3], x[4]/ref_U, x[5]/ref_U, x[6]/(ref_U/ref_L)]

    return xn

end

function unnormalize(model::Cylinder, ref_L::AbstractFloat)

    nmodel = Cylinder(model.m * (ref_L^2), model.J * (ref_L^4),
        model.cl * ref_L, model.r_b .* ref_L, model.ds .* ref_L,
        model.rhat .* ref_L, model.S, model.R, model.nodes, false)

    return nmodel

end
function unnormalize(model::Cylinder, x::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)

    xn = [x[1]*ref_L, x[2]*ref_L, x[3], x[4]*ref_U, x[5]*ref_U, x[6]*(ref_U/ref_L)]

    return xn

end

function plot_boundary(model::Cylinder, x::AbstractVector; color=:black, color2=:red)

    # Q = Rotations.RotMatrix(x[3])
    # point1_loc = x[1:2] + Q*[model.cl/2, 0]

    boundary = Circle(Point2f(x[1], x[2]), model.cl/2)
    # point1 = Circle(Point2f(point1_loc[1], point1_loc[2]), model.cl/20)

    boundary_plot = poly(boundary, color=color)
    # poly!(ax, point1, color=color2)
    
    return boundary_plot
    
end
function plot_boundary!(model::Cylinder, x::AbstractVector; color=:black, color2=:red)

    # Q = Rotations.RotMatrix(x[3])
    # point1_loc = x[1:2] + Q*[model.cl/2, 0]

    boundary = Circle(Point2f(x[1], x[2]), model.cl/2)
    # point1 = Circle(Point2f(point1_loc[1], point1_loc[2]), model.cl/20)

    boundary_plot = poly!(boundary, color=color)
    # poly!(point1, color=color2)

    return boundary_plot
    
end