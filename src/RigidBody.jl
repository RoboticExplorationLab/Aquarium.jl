"""
1D, Rigid Immersed Boundary Model
"""

abstract type RigidBody <: ImmersedBoundary end

function boundary_state(model::RigidBody, x::AbstractVector)

    # NOTE: returns x_b in INERTIAL frame

    # xn_b = ones(length(model.r_b)*2)
    # boundary_state!(model, xn_b, x)

    Q = Rotations.RotMatrix(x[3])
    R = model.R
    Q_b = R'*kron(I(model.nodes), Q)*R

    rx0_b = model.r_b[1:end÷2]
    ry0_b = model.r_b[end÷2+1:end]

    r_b = Q_b*vcat(rx0_b, ry0_b) + kron(x[1:2], ones(model.nodes))
    u_b = Q_b*(x[6] .* vcat(-ry0_b, rx0_b) + kron(x[4:5], ones(model.nodes)))

    return vcat(r_b, u_b)

end
function boundary_state!(model::RigidBody, x_b::AbstractVector, x::AbstractVector)

    # NOTE: returns x_b in INERTIAL frame

    Q = Rotations.RotMatrix(x[3])
    R = model.R
    Q_b = R'*kron(I(model.nodes), Q)*R

    rx0_b = model.r_b[1:end÷2]
    ry0_b = model.r_b[end÷2+1:end]

    r_b = Q_b*vcat(rx0_b, ry0_b) + kron(x[1:2], ones(model.nodes))
    u_b = Q_b*(x[6] .* vcat(-ry0_b, rx0_b) + kron(x[4:5], ones(model.nodes)))

    x_b .= vcat(r_b, u_b)

end

function boundary_state_jacobian(model::RigidBody, x::AbstractVector)

    # NOTE: returns x_b in INERTIAL frame

    # ∂x_b∂x = spzeros(length(model.r_b)*2, length(x))
    # boundary_state_jacobian!(model, ∂x_b∂x, x)

    rx0_b = model.r_b[1:end÷2]
    ry0_b = model.r_b[end÷2+1:end]

    Q = Rotations.RotMatrix(x[3])
    R = model.R
    Q_b = R'*kron(I(model.nodes), Q)*R

    ∂Q∂x3 = [-sin(x[3]) -cos(x[3]); cos(x[3]) -sin(x[3])]
    ∂Q_b∂x3 = model.R'*kron(I(model.nodes), ∂Q∂x3)*model.R

    ∂r_b∂x = spzeros(length(model.r_b), length(x))
    ∂r_b∂x[:, 1:2] = kron(I(2), ones(model.nodes))
    ∂r_b∂x[:, 3] = ∂Q_b∂x3*vcat(rx0_b, ry0_b)

    ∂u_b∂x = spzeros(length(model.r_b), length(x))
    ∂u_b∂x[:, 3] = ∂Q_b∂x3*(x[6] .* vcat(-ry0_b, rx0_b) + kron(x[4:5], ones(model.nodes)))
    ∂u_b∂x[:, 4:5] = Q_b*kron(I(2), ones(model.nodes))
    ∂u_b∂x[:, 6] = Q_b*vcat(-ry0_b, rx0_b)

    ∂x_b∂x = vcat(∂r_b∂x, ∂u_b∂x)

    return ∂x_b∂x

end
function boundary_state_jacobian!(model::RigidBody, ∂x_b∂x::AbstractVecOrMat, x::AbstractVector)

    rx0_b = model.r_b[1:end÷2]
    ry0_b = model.r_b[end÷2+1:end]

    Q = Rotations.RotMatrix(x[3])
    R = model.R
    Q_b = R'*kron(I(model.nodes), Q)*R
    
    ∂Q∂x3 = [-sin(x[3]) -cos(x[3]); cos(x[3]) -sin(x[3])]
    ∂Q_b∂x3 = model.R'*kron(I(model.nodes), ∂Q∂x3)*model.R

    ∂r_b∂x = spzeros(length(model.r_b), length(x))
    ∂r_b∂x[:, 1:2] = kron(I(2), ones(model.nodes))
    ∂r_b∂x[:, 3] = ∂Q_b∂x3*vcat(rx0_b, ry0_b)

    ∂u_b∂x = spzeros(length(model.r_b), length(x))
    ∂u_b∂x[:, 3] = ∂Q_b∂x3*(x[6] .* vcat(-ry0_b, rx0_b) + kron(x[4:5], ones(model.nodes)))
    ∂u_b∂x[:, 4:5] = Q_b*kron(I(2), ones(model.nodes))
    ∂u_b∂x[:, 6] = Q_b*vcat(-ry0_b, rx0_b)

    ∂x_b∂x .= vcat(∂r_b∂x, ∂u_b∂x)

end

function body_force(model::RigidBody, x::AbstractVector,
    f_b::AbstractVector, fext::AbstractVector)

    # NOTE: returns f in BODY frame. τ equivalent in both frames

    R = model.R
    S = model.S
    rhat = model.rhat

    Q = Rotations.RotMatrix(x[3])'
    Q_b = R'*kron(I(model.nodes), Q)*R
    f_b_body = Q_b*f_b
    fext_B = Q*fext

    f = S*f_b_body + fext_B
    τ = ones(model.nodes)'*rhat*f_b_body

    return f, τ

end

function body_force_jacobian(model::RigidBody, x::AbstractVector,
    f_b::AbstractVector, fext::AbstractVector)

    # ∂f∂x = zeros(3, length(x))
    # ∂f∂f_b = spzeros(3, length(f_b))

    # body_force_jacobian!(model, ∂f∂x, ∂f∂f_b, x, f_b, fext)

    R = model.R
    S = model.S
    rhat = model.rhat

    Q = Rotations.RotMatrix(x[3])'
    Q_b = R'*kron(I(model.nodes), Q)*R

    ∂Q∂x3 = [-sin(x[3]) cos(x[3]); -cos(x[3]) -sin(x[3])]
    ∂Q_b∂x3 = model.R'*kron(I(model.nodes), ∂Q∂x3)*model.R

    ∂f∂x = zeros(3, length(x))
    ∂f∂x[:, 3] = vcat(S*∂Q_b∂x3*f_b + ∂Q∂x3*fext, ones(model.nodes)'*rhat*∂Q_b∂x3*f_b)

    ∂f∂f_b = vcat(S*Q_b, ones(model.nodes)'*rhat*Q_b)

    return ∂f∂x, ∂f∂f_b

end
function body_force_jacobian!(model::RigidBody, ∂f∂x::AbstractVecOrMat, ∂f∂f_b::AbstractVecOrMat, 
    x::AbstractVector, f_b::AbstractVector, fext::AbstractVector)

    # NOTE: returns f in BODY frame. τ equivalent in both frames

    R = model.R
    S = model.S
    rhat = model.rhat

    Q = Rotations.RotMatrix(x[3])'
    Q_b = R'*kron(I(model.nodes), Q)*R

    ∂Q∂x3 = [-sin(x[3]) cos(x[3]); -cos(x[3]) -sin(x[3])]
    ∂Q_b∂x3 = model.R'*kron(I(model.nodes), ∂Q∂x3)*model.R

    ∂f∂x .= zeros(3, length(x))
    ∂f∂x[:, 3] .= vcat(S*∂Q_b∂x3*f_b + ∂Q∂x3*fext, ones(model.nodes)'*rhat*∂Q_b∂x3*f_b)

    ∂f∂f_b .= vcat(S*Q_b, ones(model.nodes)'*rhat*Q_b)

end

function dynamics(model::RigidBody, x::AbstractVector,
    f_b::AbstractVector, fext::AbstractVector=[0, 0])

    # xdot = deepcopy(x)
    # dynamics!(model, xdot, x, f_b, fext)

    r = x[1:2] # N frame
    Θ = x[3] # N frame
    v = x[4:5] # B frame
    ω = x[6] # B frame

    m = model.m
    J = model.J

    Q = Rotations.RotMatrix(Θ)

    f, τ = body_force(model, x, f_b, fext)

    xdot1 = Q*v
    xdot2 = ω
    xdot3 = f./m - ω.*[-v[2], v[1]]
    xdot4 = (1/J)*τ

    xdot = vcat(xdot1, xdot2, xdot3, xdot4)

    return xdot

end
function dynamics!(model::RigidBody, xdot::AbstractVector, x::AbstractVector,
    f_b::AbstractVector, fext::AbstractVector=[0, 0])
    
    r = x[1:2] # N frame
    Θ = x[3] # N frame
    v = x[4:5] # B frame
    ω = x[6] # B frame

    m = model.m
    J = model.J

    Q = Rotations.RotMatrix(Θ)

    f, τ = body_force(model, x, f_b, fext)

    xdot1 = Q*v
    xdot2 = ω
    xdot3 = f./m - ω.*[-v[2], v[1]]
    xdot4 = (1/J)*τ

    xdot .= vcat(xdot1, xdot2, xdot3, xdot4)

end

function dynamics_jacobian(model::RigidBody, x::AbstractVector,
    f_b::AbstractVector, fext::AbstractVector=[0, 0])

    # ∂R∂x = zeros(length(x), length(x))
    # ∂R∂f_b = spzeros(length(x), length(f_b))
    # dynamics_jacobian!(model, ∂R∂x, ∂R∂f_b, x, f_b, fext)

    r = x[1:2] # N frame
    Θ = x[3] # N frame
    v = x[4:5] # B frame
    ω = x[6] # B frame

    m = model.m
    J = model.J

    Q = Rotations.RotMatrix(Θ)

    ∂Q∂x3 = [-sin(x[3]) -cos(x[3]); cos(x[3]) -sin(x[3])]
    ∂f∂x, ∂f∂f_b = body_force_jacobian(model, x, f_b, fext)

    ∂R∂x = spzeros(length(x), length(x))
    ∂R∂x[1:2, 3] .= ∂Q∂x3*v
    ∂R∂x[1:2, 4:5] .= Q
    ∂R∂x[3, 6] = 1.0
    ∂R∂x[4:5, :] .= (1/m).*∂f∂x[1:2, :] - [0 0 0 0 -ω -v[2]; 0 0 0 ω 0 v[1]]
    ∂R∂x[6, :] .= (1/J).*∂f∂x[3, :]

    ∂R∂f_b = spzeros(length(x), length(f_b))
    ∂R∂f_b[4:5, :] .= (1/m).*∂f∂f_b[1:2, :]
    ∂R∂f_b[6, :] .= (1/J).*∂f∂f_b[3, :]

    return ∂R∂x, ∂R∂f_b

end
function dynamics_jacobian!(model::RigidBody, ∂R∂x::AbstractVecOrMat, ∂R∂f_b::AbstractVecOrMat,
    x::AbstractVector, f_b::AbstractVector, fext::AbstractVector=[0, 0])
    
    r = x[1:2] # N frame
    Θ = x[3] # N frame
    v = x[4:5] # B frame
    ω = x[6] # B frame

    m = model.m
    J = model.J

    Q = Rotations.RotMatrix(Θ)

    ∂Q∂x3 = [-sin(x[3]) -cos(x[3]); cos(x[3]) -sin(x[3])]
    ∂f∂x, ∂f∂f_b = body_force_jacobian(model, x, f_b, fext)

    ∂R∂x .= spzeros(length(x), length(x))
    ∂R∂x[1:2, 3] .= ∂Q∂x3*v
    ∂R∂x[1:2, 4:5] .= Q
    ∂R∂x[3, 6] = 1.0
    ∂R∂x[4:5, :] .= (1/m).*∂f∂x[1:2, :] - [0 0 0 0 -ω -v[2]; 0 0 0 ω 0 v[1]]
    ∂R∂x[6, :] .= (1/J).*∂f∂x[3, :]

    ∂R∂f_b .= spzeros(length(x), length(f_b))
    ∂R∂f_b[4:5, :] .= (1/m).*∂f∂f_b[1:2, :]
    ∂R∂f_b[6, :] .= (1/J).*∂f∂f_b[3, :]

end

function discrete_dynamics(model::RigidBody, fn_b::AbstractVector,
    xk::AbstractVector, fext::AbstractVector=[0, 0];
    dt::Float64=0.01, max_iter::Int64=10, tol::Float64=1e-6, verbose=false)

    xn = deepcopy(xk)
    r(x) = Aquarium.R2(model, x, fn_b, xk, fext; dt=dt)

    if verbose
        @show maximum(abs.(r(xn)))
    end

    num_iter = 0

    while maximum(abs.(r(xn))) > tol && num_iter <= max_iter

        num_iter = num_iter + 1

        R, _, _ = discrete_dynamics_jacobian(model, xn, fn_b, fext; dt=dt)

        Δxn = -R\r(xn)
        xn += Δxn

        if verbose
            @show maximum(abs.(r(xn)))
        end

    end
    
    return xn
end
function discrete_dynamics!(model::RigidBody, xn::AbstractVector, fn_b::AbstractVector,
    xk::AbstractVector, fext::AbstractVector=[0, 0];
    dt::Float64=0.01, max_iter::Int64=10, tol::Float64=1e-6, verbose=false)

    xn .= deepcopy(xk)
    r(x) = Aquarium.R2(model, x, fn_b, xk, fext; dt=dt)

    if verbose
        @show maximum(abs.(r(xn)))
    end

    num_iter = 0

    while maximum(abs.(r(xn))) > tol && num_iter <= max_iter

        num_iter = num_iter + 1

        R, _, _ = discrete_dynamics_jacobian(model, xn, fn_b, fext; dt=dt)

        Δxn = -R\r(xn)
        xn .+= Δxn

        if verbose
            @show maximum(abs.(r(xn)))
        end

    end
end

function discrete_dynamics_jacobian(model::RigidBody, xn::AbstractVector, fn_b::AbstractVector, fext::AbstractVector=[0, 0]; dt::Float64=0.01)

    # ∂R2∂xk = spzeros(length(xk), length(xk))
    # ∂R2∂xn = spzeros(length(xn), length(xn))
    # ∂R2∂fn_b = spzeros(length(xn), length(fn_b))

    # discrete_dynamics_jacobian!(model, ∂R2∂xn, ∂R2∂fn_b, ∂R2∂xk, xn, fn_b, fext; dt=dt)

    ∂R∂xn, ∂R∂fn_b = dynamics_jacobian(model, xn, fn_b, fext)

    n_x = length(xn)

    ∂R2∂xn = sparse(I, n_x, n_x) - dt.*∂R∂xn
    ∂R2∂fn_b = -dt.*∂R∂fn_b

    ∂R2∂xk = -sparse(I, n_x, n_x)

    return ∂R2∂xn, ∂R2∂fn_b, ∂R2∂xk

end
function discrete_dynamics_jacobian!(model::RigidBody, ∂R2∂xn::AbstractVecOrMat, ∂R2∂fn_b::AbstractVecOrMat,
    ∂R2∂xk::AbstractVecOrMat, xn::AbstractVector, fn_b::AbstractVector, fext::AbstractVector=[0, 0]; dt::Float64=0.01)
    
    ∂R∂xn, ∂R∂fn_b = dynamics_jacobian(model, xn, fn_b, fext)

    n_x = length(xn)

    ∂R2∂xn .= sparse(I, n_x, n_x) - dt.*∂R∂xn
    ∂R2∂fn_b .= -dt.*∂R∂fn_b

    ∂R2∂xk .= -sparse(I, n_x, n_x)

end