"""
Strongly-coupled FSI
"""

struct FSIModel
    dt::Float64 # time step
    ρ::Float64 # fluid density (kg/m^3)
    μ::Float64 # fluid dynamic viscosity (Pa*s)
    L_x::Float64 # fluid environment length (m)
    L_y::Float64 # fluid environment height (m)
    ref_L::Float64 # reference length for normalization
    ref_u::Float64 # reference length for normalization
    ne_x::Int64 # number of elements in x 
    ne_y::Int64 # number of elements in y
    h_x::Float64 # spatial step size in x 
    h_y::Float64 # spatial step size in y
    x_coord_p::Vector{Float64} # x coordinates of pressure cv
    y_coord_p::Vector{Float64} # y coordinates of pressure cv
    x_coord_ux::Vector{Float64} # x coordinates of ux velocity cv
    y_coord_ux::Vector{Float64} # y coordinates of ux velocity cv
    x_coord_uy::Vector{Float64} # x coordinates of uy velocity cv
    y_coord_uy::Vector{Float64} # y coordinates of uy velocity cv
    u_west_bc::SVector{2, <:AbstractFloat} # boundary conditions west (m/s)
    u_east_bc::SVector{2, <:AbstractFloat} # boundary conditions east (m/s)
    u_north_bc::SVector{2, <:AbstractFloat} # boundary conditions north (m/s)
    u_south_bc::SVector{2, <:AbstractFloat} # boundary conditions south (m/s)
    outflow::SVector{4, Bool} # outflow boundary conditions
    Re::Float64 # Reynolds number
    FVM_ops::FVM_CDS_2D # FVM discretized ops
    normalize::Bool # are parameters normalized
end

function FSIModel(dt::AbstractFloat=0.01, ρ::AbstractFloat=997.0, μ::AbstractFloat=8.9e-4,
    L_x::AbstractFloat=1.0, L_y::AbstractFloat=1.0, ref_L::AbstractFloat=1.0,
    ref_u::AbstractFloat=0.0, ne_x::Int=20, ne_y::Int=20,
    u_west_bc::SVector{2, <:AbstractFloat}=SA[0.0, 0.0],
    u_east_bc::SVector{2, <:AbstractFloat}=SA[0.0, 0.0],
    u_north_bc::SVector{2, <:AbstractFloat}=SA[0.0, 0.0],
    u_south_bc::SVector{2, <:AbstractFloat}=SA[0.0, 0.0],
    outflow::SVector{4, Bool}=SA[false, false, false, false];
    normalize=true)
    
    # combine boundary conditions
    bc_vel = [u_west_bc, u_east_bc, u_north_bc, u_south_bc]

    # calculate spatial stepsizes
    h_x = L_x/ne_x
    h_y = L_y/ne_y
    
    # calculate Reynolds #
    Re = ref_u*ref_L/(μ/ρ)

    # FVM discretization
    FVM_ops = FVM_CDS_2D(SA[ne_x, ne_y], SA[h_x, h_y], bc_vel, outflow)

    # define positions of pressure cv
    x_coord_p = LinRange(h_x/2, L_x-h_x/2, ne_x)
    y_coord_p = LinRange(h_y/2, L_y-h_y/2, ne_y)

    # define coordinates of u cv
    x_coord_ux = LinRange(h_x, L_x-h_x, ne_x-1)
    y_coord_ux = LinRange(h_y/2, L_y-h_y/2, ne_y)
    
    x_coord_ux, y_coord_ux = meshgrid_vec(x_coord_ux, y_coord_ux)

    # define coordinates of v cv
    x_coord_uy = LinRange(h_x/2, L_x-h_x/2, ne_x)
    y_coord_uy = LinRange(h_y, L_y-h_y, ne_y-1)

    x_coord_uy, y_coord_uy = meshgrid_vec(x_coord_uy, y_coord_uy)
    
    # create FSIModel
    model = FSIModel(dt, ρ, μ, L_x, L_y, ref_L, ref_u, ne_x, ne_y, h_x, h_y,
        x_coord_p, y_coord_p, x_coord_ux, y_coord_ux, x_coord_uy, y_coord_uy,
        u_west_bc, u_east_bc, u_north_bc, u_south_bc, outflow, Re, FVM_ops, normalize
    )
    
    # normalize
    if normalize
        model = normalize(model)
    end

    return model
end

function normalize(model::FSIModel)

    # extract reference length and velocity value
    ref_L = model.ref_L
    ref_u = model.ref_u

    # define normalized cavity dimensions
    L_x = model.L_x / ref_L
    L_y = model.L_y / ref_L

    h_x = L_x / model.ne_x
    h_y = L_y / model.ne_y

    # define normalized boundary conditions
    u_west_bc = model.u_west_bc ./ ref_u
    u_east_bc = model.u_east_bc ./ ref_u
    u_north_bc = model.u_north_bc ./ ref_u
    u_south_bc = model.u_south_bc ./ ref_u

    # combine boundary conditions
    bc_vel = [u_west_bc, u_east_bc, u_north_bc, u_south_bc]

    # FVM discretization
    FVM_ops = FVM_CDS_2D(SA[model.ne_x, model.ne_y], SA[h_x, h_y],
        bc_vel, model.outflow)

    # define normalized positions of pressure cv
    x_coord_p = LinRange(h_x/2, L_x-h_x/2, model.ne_x)
    y_coord_p = LinRange(h_y/2, L_y-h_y/2, model.ne_y)

    # define normalized coordinates of u cv
    x_coord_ux = LinRange(h_x, L_x-h_x, model.ne_x-1)
    y_coord_ux = LinRange(h_y/2, L_y-h_y/2, model.ne_y)

    x_coord_ux, y_coord_ux = meshgrid_vec(x_coord_ux, y_coord_ux)
        
    # define normalized coordinates of v cv
    x_coord_uy = LinRange(h_x/2, L_x-h_x/2, model.ne_x)
    y_coord_uy = LinRange(h_y, L_y-h_y, model.ne_y-1)

    x_coord_uy, y_coord_uy = meshgrid_vec(x_coord_uy, y_coord_uy)

    # define normalized time properties
    dt = model.dt / (ref_L/ref_u)

    # set normalize to true
    normalize = true
    
    # create FSIModel
    nmodel = FSIModel(dt, model.ρ, model.μ, L_x, L_y, ref_L, ref_u,
        model.ne_x, model.ne_y, h_x, h_y, x_coord_p, y_coord_p,
        x_coord_ux, y_coord_ux, x_coord_uy, y_coord_uy,
        u_west_bc, u_east_bc, u_north_bc, u_south_bc,
        model.outflow, model.Re, FVM_ops, normalize
    )

    return nmodel
end
function unnormalize(nmodel::FSIModel)

    # extract reference length and velocity value
    ref_L = nmodel.ref_L
    ref_u = nmodel.ref_u

    # define normalized cavity dimensions
    L_x = nmodel.L_x * ref_L
    L_y = nmodel.L_y * ref_L

    h_x = L_x / nmodel.ne_x
    h_y = L_y / nmodel.ne_y

    # define normalized boundary conditions
    u_west_bc = nmodel.u_west_bc .* ref_u
    u_east_bc = nmodel.u_east_bc .* ref_u
    u_north_bc = nmodel.u_north_bc .* ref_u
    u_south_bc = nmodel.u_south_bc .* ref_u

    # combine boundary conditions
    bc_vel = [u_west_bc, u_east_bc, u_north_bc, u_south_bc]

    # FVM discretization
    FVM_ops = FVM_CDS_2D(SA[nmodel.ne_x, nmodel.ne_y], SA[h_x, h_y],
        bc_vel, nmodel.outflow)

    # define unnormalized positions of pressure cv
    x_coord_p = LinRange(h_x/2, L_x-h_x/2, nmodel.ne_x)
    y_coord_p = LinRange(h_y/2, L_y-h_y/2, nmodel.ne_y)

    # define unnormalized coordinates of u cv
    x_coord_ux = LinRange(h_x, L_x-h_x, nmodel.ne_x-1)
    y_coord_ux = LinRange(h_y/2, L_y-h_y/2, nmodel.ne_y)
    
    x_coord_ux, y_coord_ux = meshgrid_vec(x_coord_ux, y_coord_ux)

    # define unnormalized coordinates of v cv
    x_coord_uy = LinRange(h_x/2, L_x-h_x/2, nmodel.ne_x)
    y_coord_uy = LinRange(h_y, L_y-h_y, nmodel.ne_y-1)

    x_coord_uy, y_coord_uy = meshgrid_vec(x_coord_uy, y_coord_uy)

    # define normalized time properties
    dt = nmodel.dt * (ref_L/ref_u)

    # set normalize to false
    normalize = false
    
    # create FSIModel
    model = FSIModel(dt, nmodel.ρ, nmodel.μ, L_x, L_y, ref_L, ref_u,
        nmodel.ne_x, nmodel.ne_y, h_x, h_y, x_coord_p, y_coord_p,
        x_coord_ux, y_coord_ux, x_coord_uy, y_coord_uy,
        u_west_bc, u_east_bc, u_north_bc, u_south_bc,
        nmodel.outflow, nmodel.Re, FVM_ops, normalize
    )

    return model
end

function initialize(model::FSIModel, boundary::ImmersedBoundary)

    # extract bc
    u_west_bc = model.u_west_bc
    u_east_bc = model.u_east_bc
    u_north_bc = model.u_north_bc
    u_south_bc = model.u_south_bc

    # extract outflow booleans
    outflow_w, outflow_e, outflow_n, outflow_s = model.outflow

    # extract element #
    ne_x = model.ne_x
    ne_y = model.ne_y

    # number of cv
    n_ux = (ne_x-1)*ne_y
    n_uy = ne_x*(ne_y-1)
    n_p = ne_x*ne_y
    
    n_b = boundary.nodes

    # initialize velocity, pressure, and force states
    uk = zeros(n_ux + n_uy)
    pk = zeros(n_p)
    fk_tilda = zeros(2*n_b)

    if u_west_bc == u_east_bc
        uk[1:n_ux] .= u_west_bc[1]
        uk[n_ux+1:end] .= u_west_bc[2]
    elseif u_north_bc == u_south_bc
        uk[1:n_ux] .= u_north_bc[1]
        uk[n_ux+1:end] .= u_north_bc[2]
    end

    if outflow_e
        uk[1:n_ux] .= u_west_bc[1]
        uk[n_ux+1:end] .= u_west_bc[2]
    elseif outflow_w
        uk[1:n_ux] .= u_east_bc[1]
        uk[n_ux+1:end] .= u_east_bc[2]
    elseif outflow_n
        uk[1:n_ux] .= u_south_bc[1]
        uk[n_ux+1:end] .= u_south_bc[2]
    elseif outflow_s
        uk[1:n_ux] .= u_north_bc[1]
        uk[n_ux+1:end] .= u_north_bc[2]
    end

    return uk, pk, fk_tilda

end

include(joinpath(@__DIR__, "FSI_weakly_coupled_dynamics.jl"))

function R1(model::FSIModel, boundary::ImmersedBoundary,
    uk::AbstractVector, pk::AbstractVector, fk_b::AbstractVector,
    xk::AbstractVector, ukm1::AbstractVector)

    ds = vcat(boundary.ds, boundary.ds)

    E = boundary_coupling(model, boundary, boundary_state(boundary, xk))

    fk_b_tilda = fk_b ./ (model.ρ * model.h_x * model.h_y)
    
    ds = vcat(boundary.ds, boundary.ds)

    # extract operators
    G = model.FVM_ops.G
    L = model.FVM_ops.L[1]
    L_bc = model.FVM_ops.L[3]

    dt = model.dt
    Re = model.Re

    nf_u = size(L, 1)

    # define kkt system submatrices
    A = (1/dt).*(sparse(I, nf_u, nf_u) - (dt/(2*Re)).*L)
    r = (1/dt).*(sparse(I, nf_u, nf_u) + (dt/(2*Re)).*L)*ukm1 - 0.5.*N(model, ukm1)
    bc1 = (1/Re).*L_bc
    
    return A*uk + 0.5.*N(model, uk) - (r + bc1) + G*pk + E'*fk_b_tilda
end
function c1(model::FSIModel, uk::AbstractVector)

    G = model.FVM_ops.G
    D_bc = model.FVM_ops.D[2]
    bc2 = -D_bc
    
    return G'*uk + bc2
end
function c2(model::FSIModel, boundary::ImmersedBoundary,
    uk::AbstractVector, xk::AbstractVector)

    xk_b = boundary_state(boundary, xk)

    return boundary_coupling(model, boundary, xk_b)*uk - xk_b[end÷2+1:end]
end

function N(model::FSIModel, u::AbstractVector)

    m1, m2, m3, m4, m5, m6 = model.FVM_ops.N[1]
    m1_bc, m2_bc, m3_bc, m4_bc, m5_bc, m6_bc = model.FVM_ops.N[2]
    
    N1 = m1*u + m1_bc
    N2 = m2*u + m2_bc
    N3 = m3*u + m3_bc
    N4 = m4*u + m4_bc
    N5 = m5*u + m5_bc
    N6 = m6*u + m6_bc
    
    convective = (N1.*N1 - N2.*N2) + (N3.*N4 - N5.*N6)
    return convective
    
end
function N_jacobian(model::FSIModel, u::AbstractVector)

    m1, m2, m3, m4, m5, m6 = model.FVM_ops.N[1]
    m1_bc, m2_bc, m3_bc, m4_bc, m5_bc, m6_bc = model.FVM_ops.N[2]
    
    N1 = m1*u + m1_bc
    N2 = m2*u + m2_bc
    N3 = m3*u + m3_bc
    N4 = m4*u + m4_bc
    N5 = m5*u + m5_bc
    N6 = m6*u + m6_bc
        
    ∂N∂u = (spdiagm(N1)*m1.*2 - spdiagm(N2)*m2.*2) + 
        (spdiagm(N3)*m4 + spdiagm(N4)*m3) -
        (spdiagm(N5)*m6 + spdiagm(N6)*m5)
    return ∂N∂u

end

function discrete_dynamics_jacobian(model::FSIModel, boundary::ImmersedBoundary,
    uk::AbstractVector, fk_b::AbstractVector, xk::AbstractVector, ukm1::AbstractVector)

    ds = vcat(boundary.ds, boundary.ds)
    fk_b_tilda = fk_b ./ (model.ρ * model.h_x * model.h_y)

    xk_b = boundary_state(boundary, xk)
    E, ∂vecE_∂xk_b, ∂vecE_T∂xk_b = boundary_coupling_with_jacobian(model, boundary, xk_b)
    G = model.FVM_ops.G
    L = model.FVM_ops.L[1]

    dt = model.dt
    Re = model.Re

    nf_u = size(L, 1)
    n_b = Int(length(xk_b)/4)

    # define kkt system submatrices
    A = (1/dt).*(sparse(I, nf_u, nf_u) - (dt/(2*Re)).*L)

    ∂xk_b∂xk = boundary_state_jacobian(boundary, xk)

    ∂R1∂uk = A + 0.5.*N_jacobian(model, uk)
    ∂R1∂ukm1 = -(1/dt).*(sparse(I, nf_u, nf_u) + (dt/(2*Re)).*L) + 0.5.*N_jacobian(model, ukm1)
    ∂R1∂pk = G
    ∂R1∂fk_b = sparse(E') ./ (model.ρ * model.h_x * model.h_y)
    ∂R1∂xk_b = kron(sparse(fk_b_tilda'), sparse(I, nf_u, nf_u))*∂vecE_T∂xk_b
    ∂R1∂xk = ∂R1∂xk_b*∂xk_b∂xk
    
    ∂c1∂uk = sparse(G')
    
    ∂c2∂xk_b = kron(sparse(uk'), sparse(I, n_b*2, n_b*2))*∂vecE_∂xk_b - kron(sparse([0, 1]'), sparse(I, n_b*2, n_b*2))
    ∂c2∂xk = ∂c2∂xk_b*∂xk_b∂xk
    ∂c2∂uk = E

    return ∂R1∂uk, ∂R1∂ukm1, ∂R1∂pk, ∂R1∂fk_b, ∂R1∂xk_b, ∂R1∂xk, ∂c1∂uk, ∂c2∂uk, ∂c2∂xk_b, ∂c2∂xk

end

function boundary_coupling(model::FSIModel, boundary::ImmersedBoundary, x_b::AbstractVector)

    rx_b = x_b[1:boundary.nodes]
    ry_b = x_b[boundary.nodes+1:2*boundary.nodes]
    
    x_coord_ux = model.x_coord_ux
    y_coord_ux = model.y_coord_ux
    
    x_coord_uy = model.x_coord_uy
    y_coord_uy = model.y_coord_uy
    
    # extract spatial interval steps
    h_x = model.h_x
    h_y = model.h_y
    
    # determine number of eulerian and lagrange coordinates
    n_ux = length(x_coord_ux);
    n_uy = length(x_coord_uy);
    n_b = length(rx_b);
    
    # create coordinate "matrices" for input into discrete delta functions
    ux_x = (ones(1, n_b) .* x_coord_ux)'
    ux_y = (ones(1, n_b) .* y_coord_ux)'
    
    b_ux_x = ones(1, n_ux) .* rx_b
    b_ux_y = ones(1, n_ux) .* ry_b
    
    uy_x = (ones(1, n_b) .* x_coord_uy)'
    uy_y = (ones(1, n_b) .* y_coord_uy)'
    
    b_uy_x = ones(1, n_uy) .* rx_b
    b_uy_y = ones(1, n_uy) .* ry_b
        
    # calculate delta function values for x and y
    
    Dux_x = discrete_delta((ux_x - b_ux_x) ./ h_x)
    Dux_y = discrete_delta((ux_y - b_ux_y) ./ h_y)
    
    Duy_x = discrete_delta((uy_x - b_uy_x) ./ h_x)
    Duy_y = discrete_delta((uy_y - b_uy_y) ./ h_y)
    
    # calculate all kernel values corresponding to boundary nodes
    E_u = Dux_x .* Dux_y
    E_v = Duy_x .* Duy_y
    
    E = blockdiag(E_u, E_v)

    return E

end
function boundary_coupling_jacobian(model::FSIModel, boundary::ImmersedBoundary, x_b::AbstractVector)

    rx_b = x_b[1:boundary.nodes]
    ry_b = x_b[boundary.nodes+1:2*boundary.nodes]

    x_coord_ux = model.x_coord_ux
    y_coord_ux = model.y_coord_ux

    x_coord_uy = model.x_coord_uy
    y_coord_uy = model.y_coord_uy

    # extract spatial interval steps
    h_x = model.h_x
    h_y = model.h_y

    # determine number of eulerian and lagrange coordinates
    n_ux = length(x_coord_ux)
    n_uy = length(x_coord_uy)
    n_b = length(rx_b)

    # create coordinate "matrices" for input into discrete delta functions
    ux_x = (ones(1, n_b) .* x_coord_ux)'
    ux_y = (ones(1, n_b) .* y_coord_ux)'
    
    b_ux_x = ones(1, n_ux) .* rx_b
    b_ux_y = ones(1, n_ux) .* ry_b
    
    uy_x = (ones(1, n_b) .* x_coord_uy)'
    uy_y = (ones(1, n_b) .* y_coord_uy)'
    
    b_uy_x = ones(1, n_uy) .* rx_b
    b_uy_y = ones(1, n_uy) .* ry_b
        
    # calculate delta function values for x and y

    Dux_x = discrete_delta((ux_x - b_ux_x) ./ h_x)
    Dux_y = discrete_delta((ux_y - b_ux_y) ./ h_y)

    Duy_x = discrete_delta((uy_x - b_uy_x) ./ h_x)
    Duy_y = discrete_delta((uy_y - b_uy_y) ./ h_y)

    ∂Dux∂rx_b = -discrete_delta_jacobian((ux_x - b_ux_x) ./ h_x) ./ h_x
    ∂Dux∂ry_b = -discrete_delta_jacobian((ux_y - b_ux_y) ./ h_y) ./ h_y

    ∂Duy∂rx_b = -discrete_delta_jacobian((uy_x - b_uy_x) ./ h_x) ./ h_x
    ∂Duy∂ry_b = -discrete_delta_jacobian((uy_y - b_uy_y) ./ h_y) ./ h_y

    if maximum([n_ux, n_uy]) >= 128

        I_ux = SharedVector{Int64}(n_ux*2*n_b)
        J_ux = SharedVector{Int64}(n_ux*2*n_b)
        values_ux = SharedVector{Float64}(n_ux*2*n_b)

        I_uy = SharedVector{Int64}(n_uy*2*n_b)
        J_uy = SharedVector{Int64}(n_uy*2*n_b)
        values_uy = SharedVector{Float64}(n_uy*2*n_b)
        
        @threads for i in 1:maximum([n_ux, n_uy])

            if i <= n_ux 
                
                Dux_x_i = Dux_x[:, i]
                Dux_y_i = Dux_y[:, i]
            
                ∂Dux∂rx_b_i = ∂Dux∂rx_b[:, i]
                ∂Dux∂ry_b_i = ∂Dux∂ry_b[:, i]
                
                if ~iszero(Dux_x_i) || ~iszero(Dux_y_i)

                    ∂vecE_ux∂rx_b_i = ∂Dux∂rx_b_i .* Dux_y_i
                    ∂vecE_ux∂ry_b_i = Dux_x_i .* ∂Dux∂ry_b_i

                    I_ux[1 + (i-1)*2*n_b:2*n_b*i] = kron([1, 1], (1 + n_b*2*(i-1)):(n_b*2*(i-1) + n_b))
                    J_ux[1 + (i-1)*2*n_b:2*n_b*i] = 1:2*n_b
                    values_ux[1 + (i-1)*2*n_b:2*n_b*i] = vcat(∂vecE_ux∂rx_b_i, ∂vecE_ux∂ry_b_i)

                end

            end

            if i <= n_uy

                Duy_x_i = Duy_x[:, i]
                Duy_y_i = Duy_y[:, i]
            
                ∂Duy∂rx_b_i = ∂Duy∂rx_b[:, i]
                ∂Duy∂ry_b_i = ∂Duy∂ry_b[:, i]
            
                if ~iszero(Duy_x_i) || ~iszero(Duy_y_i)
            
                    ∂vecE_uy∂rx_b_i = ∂Duy∂rx_b_i .* Duy_y_i
                    ∂vecE_uy∂ry_b_i = Duy_x_i .* ∂Duy∂ry_b_i

                    I_uy[1 + (i-1)*2*n_b:2*n_b*i] = kron([1, 1], 1 + n_b + n_b*2*(i-1):2*n_b + n_b*2*(i-1))
                    J_uy[1 + (i-1)*2*n_b:2*n_b*i] = 1:2*n_b
                    values_uy[1 + (i-1)*2*n_b:2*n_b*i] = vcat(∂vecE_uy∂rx_b_i, ∂vecE_uy∂ry_b_i)

                end

            end

        end

        nonzeros_ux = findall(!iszero, vec(values_ux))
        nonzeros_uy = findall(!iszero, vec(values_uy))

        ∂vecE_ux∂x_b = sparse(vec(I_ux)[nonzeros_ux], vec(J_ux)[nonzeros_ux], vec(values_ux)[nonzeros_ux], n_ux*n_b*2, n_b*4)
        ∂vecE_uy∂x_b = sparse(vec(I_uy)[nonzeros_uy], vec(J_uy)[nonzeros_uy], vec(values_uy)[nonzeros_uy], n_uy*n_b*2, n_b*4)
        ∂vecE∂x_b = vcat(∂vecE_ux∂x_b, ∂vecE_uy∂x_b)

    else

        ∂vecE_ux∂x_b = spzeros(n_ux*n_b*2, n_b*4)
        ∂vecE_uy∂x_b = spzeros(n_uy*n_b*2, n_b*4)
        
        for i in 1:n_ux

            Dux_x_i = Dux_x[:, i]
            Dux_y_i = Dux_y[:, i]
        
            Duy_x_i = Duy_x[:, i]
            Duy_y_i = Duy_y[:, i]
        
            ∂Dux∂rx_b_i = ∂Dux∂rx_b[:, i]
            ∂Dux∂ry_b_i = ∂Dux∂ry_b[:, i]
        
            ∂Duy∂rx_b_i = ∂Duy∂rx_b[:, i]
            ∂Duy∂ry_b_i = ∂Duy∂ry_b[:, i]

            ∂vecE_ux∂rx_b_i = ∂Dux∂rx_b_i .* Dux_y_i
            ∂vecE_ux∂ry_b_i = Dux_x_i .* ∂Dux∂ry_b_i
        
            ∂vecE_uy∂rx_b_i = ∂Duy∂rx_b_i .* Duy_y_i
            ∂vecE_uy∂ry_b_i = Duy_x_i .* ∂Duy∂ry_b_i

            ∂vecE_ux∂x_b[1 + n_b*2*(i-1):n_b*2*(i-1) + n_b, 1:end÷2] = hcat(spdiagm(∂vecE_ux∂rx_b_i), spdiagm(∂vecE_ux∂ry_b_i))
            ∂vecE_uy∂x_b[1 + n_b + n_b*2*(i-1):2*n_b + n_b*2*(i-1), 1:end÷2] = hcat(spdiagm(∂vecE_uy∂rx_b_i), spdiagm(∂vecE_uy∂ry_b_i))
        
        end

        ∂vecE∂x_b = vcat(∂vecE_ux∂x_b, ∂vecE_uy∂x_b)

    end

    return ∂vecE∂x_b
        
end
function boundary_coupling_transpose_jacobian(model::FSIModel, boundary::ImmersedBoundary, x_b::AbstractVector)

    rx_b = x_b[1:boundary.nodes]
    ry_b = x_b[boundary.nodes+1:2*boundary.nodes]

    x_coord_ux = model.x_coord_ux
    y_coord_ux = model.y_coord_ux

    x_coord_uy = model.x_coord_uy
    y_coord_uy = model.y_coord_uy

    # extract spatial interval steps
    h_x = model.h_x
    h_y = model.h_y

    # determine number of eulerian and lagrange coordinates
    n_ux = length(x_coord_ux)
    n_uy = length(x_coord_uy)
    n_b = length(rx_b)

    # create TRAR1POSES of coordinate "matrices" for input into discrete delta functions
    ux_x = ones(1, n_b) .* x_coord_ux
    ux_y = ones(1, n_b) .* y_coord_ux

    b_ux_x = (ones(1, n_ux) .* rx_b)'
    b_ux_y = (ones(1, n_ux) .* ry_b)'

    uy_x = ones(1, n_b) .* x_coord_uy
    uy_y = ones(1, n_b) .* y_coord_uy

    b_uy_x = (ones(1, n_uy) .* rx_b)'
    b_uy_y = (ones(1, n_uy) .* ry_b)'
        
    # calculate delta function values for x and y

    Dux_x = discrete_delta((ux_x - b_ux_x) ./ h_x)
    Dux_y = discrete_delta((ux_y - b_ux_y) ./ h_y)

    Duy_x = discrete_delta((uy_x - b_uy_x) ./ h_x)
    Duy_y = discrete_delta((uy_y - b_uy_y) ./ h_y)

    ∂Dux∂rx_b = -discrete_delta_jacobian((ux_x - b_ux_x) ./ h_x) ./ h_x
    ∂Dux∂ry_b = -discrete_delta_jacobian((ux_y - b_ux_y) ./ h_y) ./ h_y

    ∂Duy∂rx_b = -discrete_delta_jacobian((uy_x - b_uy_x) ./ h_x) ./ h_x
    ∂Duy∂ry_b = -discrete_delta_jacobian((uy_y - b_uy_y) ./ h_y) ./ h_y

    if n_b >= 128

        I_ux = SharedVector{Int64}(n_ux*2*n_b)
        J_ux = SharedVector{Int64}(n_ux*2*n_b)
        values_ux = SharedVector{Float64}(n_ux*2*n_b)

        I_uy = SharedVector{Int64}(n_uy*2*n_b)
        J_uy = SharedVector{Int64}(n_ux*2*n_b)
        values_uy = SharedVector{Float64}(n_uy*2*n_b)

        @threads for i in 1:boundary.nodes

            Dux_x_i = Dux_x[:, i]
            Dux_y_i = Dux_y[:, i]

            Duy_x_i = Duy_x[:, i]
            Duy_y_i = Duy_y[:, i]

            ∂Dux∂rx_b_i = ∂Dux∂rx_b[:, i]
            ∂Dux∂ry_b_i = ∂Dux∂ry_b[:, i]

            ∂Duy∂rx_b_i = ∂Duy∂rx_b[:, i]
            ∂Duy∂ry_b_i = ∂Duy∂ry_b[:, i]

            if ~iszero(Dux_x_i) || ~iszero(Dux_y_i)

                ∂vecE_ux∂rx_b_i = ∂Dux∂rx_b_i .* Dux_y_i
                ∂vecE_ux∂ry_b_i = Dux_x_i .* ∂Dux∂ry_b_i

                I_ux[1 + (i-1)*2*n_ux:2*n_ux*i] = kron([1, 1], 1 + n_ux*2*(i-1):n_ux*(2*(i-1)+1))
                J_ux[1 + (i-1)*2*n_ux:2*n_ux*i] = vcat(i.*ones(n_ux), (i+n_b).*ones(n_ux))
                values_ux[1 + (i-1)*2*n_ux:2*n_ux*i] = vcat(∂vecE_ux∂rx_b_i, ∂vecE_ux∂ry_b_i)

            end

            if ~iszero(Duy_x_i) || ~iszero(Duy_y_i)
            
                ∂vecE_uy∂rx_b_i = ∂Duy∂rx_b_i .* Duy_y_i
                ∂vecE_uy∂ry_b_i = Duy_x_i .* ∂Duy∂ry_b_i
                
                I_uy[1 + (i-1)*2*n_uy:2*n_uy*i] = kron([1, 1], 1 + n_uy*(2*(i-1)+1):n_uy*2*(i))
                J_uy[1 + (i-1)*2*n_uy:2*n_uy*i] = vcat(i.*ones(n_uy), (i+n_b).*ones(n_uy))
                values_uy[1 + (i-1)*2*n_uy:2*n_uy*i] = vcat(∂vecE_uy∂rx_b_i, ∂vecE_uy∂ry_b_i)
            
            end

        end

        nonzeros_ux = findall(!iszero, vec(values_ux))
        nonzeros_uy = findall(!iszero, vec(values_uy))

        ∂vecE_T_ux∂x_b = sparse(vec(I_ux)[nonzeros_ux], vec(J_ux)[nonzeros_ux], vec(values_ux)[nonzeros_ux], n_ux*n_b*2, n_b*4)
        ∂vecE_T_uy∂x_b = sparse(vec(I_uy)[nonzeros_uy], vec(J_uy)[nonzeros_uy], vec(values_uy)[nonzeros_uy], n_uy*n_b*2, n_b*4)
        ∂vecE_T∂x_b = vcat(∂vecE_T_ux∂x_b, ∂vecE_T_uy∂x_b)

    else

        ∂vecE_T_ux∂x_b = spzeros(n_ux*n_b*2, n_b*4)
        ∂vecE_T_uy∂x_b = spzeros(n_uy*n_b*2, n_b*4)
        
        for i in 1:boundary.nodes

            Dux_x_i = Dux_x[:, i]
            Dux_y_i = Dux_y[:, i]
    
            Duy_x_i = Duy_x[:, i]
            Duy_y_i = Duy_y[:, i]
    
            ∂Dux∂rx_b_i = ∂Dux∂rx_b[:, i]
            ∂Dux∂ry_b_i = ∂Dux∂ry_b[:, i]
    
            ∂Duy∂rx_b_i = ∂Duy∂rx_b[:, i]
            ∂Duy∂ry_b_i = ∂Duy∂ry_b[:, i]
    
            ∂vecE_ux∂rx_b_i = ∂Dux∂rx_b_i .* Dux_y_i
            ∂vecE_ux∂ry_b_i = Dux_x_i .* ∂Dux∂ry_b_i
    
            ∂vecE_uy∂rx_b_i = ∂Duy∂rx_b_i .* Duy_y_i
            ∂vecE_uy∂ry_b_i = Duy_x_i .* ∂Duy∂ry_b_i
    
            ∂vecE_T_ux∂x_b[1 + n_ux*2*(i-1):n_ux*(2*(i-1)+1), i] = ∂vecE_ux∂rx_b_i
            ∂vecE_T_ux∂x_b[1 + n_ux*2*(i-1):n_ux*(2*(i-1)+1), i+n_b] = ∂vecE_ux∂ry_b_i
    
            ∂vecE_T_uy∂x_b[1 + n_uy*(2*(i-1)+1):n_uy*2*(i), i] = ∂vecE_uy∂rx_b_i
            ∂vecE_T_uy∂x_b[1 + n_uy*(2*(i-1)+1):n_uy*2*(i), i+n_b] = ∂vecE_uy∂ry_b_i
    
        end
    
        ∂vecE_T∂x_b = vcat(∂vecE_T_ux∂x_b, ∂vecE_T_uy∂x_b)
    end

    return ∂vecE_T∂x_b
        
end
function boundary_coupling_with_jacobian(model::FSIModel, boundary::ImmersedBoundary, x_b::AbstractVector)

    rx_b = x_b[1:boundary.nodes]
    ry_b = x_b[boundary.nodes+1:2*boundary.nodes]
    
    x_coord_ux = model.x_coord_ux
    y_coord_ux = model.y_coord_ux
    
    x_coord_uy = model.x_coord_uy
    y_coord_uy = model.y_coord_uy
    
    # extract spatial interval steps
    h_x = model.h_x
    h_y = model.h_y
    
    # determine number of eulerian and lagrange coordinates
    n_ux = length(x_coord_ux);
    n_uy = length(x_coord_uy);
    n_b = length(rx_b);
    
    # create coordinate "matrices" for input into discrete delta functions
    ux_x = (ones(1, n_b) .* x_coord_ux)'
    ux_y = (ones(1, n_b) .* y_coord_ux)'
    
    b_ux_x = ones(1, n_ux) .* rx_b
    b_ux_y = ones(1, n_ux) .* ry_b
    
    uy_x = (ones(1, n_b) .* x_coord_uy)'
    uy_y = (ones(1, n_b) .* y_coord_uy)'
    
    b_uy_x = ones(1, n_uy) .* rx_b
    b_uy_y = ones(1, n_uy) .* ry_b
        
    # calculate delta function values for x and y
    
    Dux_x = discrete_delta((ux_x - b_ux_x) ./ h_x)
    Dux_y = discrete_delta((ux_y - b_ux_y) ./ h_y)
    
    Duy_x = discrete_delta((uy_x - b_uy_x) ./ h_x)
    Duy_y = discrete_delta((uy_y - b_uy_y) ./ h_y)

    ∂Dux∂rx_b = -discrete_delta_jacobian((ux_x - b_ux_x) ./ h_x) ./ h_x
    ∂Dux∂ry_b = -discrete_delta_jacobian((ux_y - b_ux_y) ./ h_y) ./ h_y

    ∂Duy∂rx_b = -discrete_delta_jacobian((uy_x - b_uy_x) ./ h_x) ./ h_x
    ∂Duy∂ry_b = -discrete_delta_jacobian((uy_y - b_uy_y) ./ h_y) ./ h_y

    if maximum([n_ux, n_uy]) >= 128

        I_ux = SharedVector{Int64}(n_ux*2*n_b)
        J_ux = SharedVector{Int64}(n_ux*2*n_b)
        values_ux = SharedVector{Float64}(n_ux*2*n_b)

        I_uy = SharedVector{Int64}(n_uy*2*n_b)
        J_uy = SharedVector{Int64}(n_uy*2*n_b)
        values_uy = SharedVector{Float64}(n_uy*2*n_b)
        
        @threads for i in 1:maximum([n_ux, n_uy])

            if i <= n_ux 
                
                Dux_x_i = Dux_x[:, i]
                Dux_y_i = Dux_y[:, i]
            
                ∂Dux∂rx_b_i = ∂Dux∂rx_b[:, i]
                ∂Dux∂ry_b_i = ∂Dux∂ry_b[:, i]
                
                if ~iszero(Dux_x_i) || ~iszero(Dux_y_i)

                    ∂vecE_ux∂rx_b_i = ∂Dux∂rx_b_i .* Dux_y_i
                    ∂vecE_ux∂ry_b_i = Dux_x_i .* ∂Dux∂ry_b_i

                    I_ux[1 + (i-1)*2*n_b:2*n_b*i] = kron([1, 1], (1 + n_b*2*(i-1)):(n_b*2*(i-1) + n_b))
                    J_ux[1 + (i-1)*2*n_b:2*n_b*i] = 1:2*n_b
                    values_ux[1 + (i-1)*2*n_b:2*n_b*i] = vcat(∂vecE_ux∂rx_b_i, ∂vecE_ux∂ry_b_i)

                end

            end

            if i <= n_uy

                Duy_x_i = Duy_x[:, i]
                Duy_y_i = Duy_y[:, i]
            
                ∂Duy∂rx_b_i = ∂Duy∂rx_b[:, i]
                ∂Duy∂ry_b_i = ∂Duy∂ry_b[:, i]
            
                if ~iszero(Duy_x_i) || ~iszero(Duy_y_i)
            
                    ∂vecE_uy∂rx_b_i = ∂Duy∂rx_b_i .* Duy_y_i
                    ∂vecE_uy∂ry_b_i = Duy_x_i .* ∂Duy∂ry_b_i

                    I_uy[1 + (i-1)*2*n_b:2*n_b*i] = kron([1, 1], 1 + n_b + n_b*2*(i-1):2*n_b + n_b*2*(i-1))
                    J_uy[1 + (i-1)*2*n_b:2*n_b*i] = 1:2*n_b
                    values_uy[1 + (i-1)*2*n_b:2*n_b*i] = vcat(∂vecE_uy∂rx_b_i, ∂vecE_uy∂ry_b_i)

                end

            end

        end

        nonzeros_ux = findall(!iszero, vec(values_ux))
        nonzeros_uy = findall(!iszero, vec(values_uy))

        ∂vecE_ux∂x_b = sparse(vec(I_ux)[nonzeros_ux], vec(J_ux)[nonzeros_ux], vec(values_ux)[nonzeros_ux], n_ux*n_b*2, n_b*4)
        ∂vecE_uy∂x_b = sparse(vec(I_uy)[nonzeros_uy], vec(J_uy)[nonzeros_uy], vec(values_uy)[nonzeros_uy], n_uy*n_b*2, n_b*4)
        ∂vecE∂x_b = vcat(∂vecE_ux∂x_b, ∂vecE_uy∂x_b)

    else

        ∂vecE_ux∂x_b = spzeros(n_ux*n_b*2, n_b*4)
        ∂vecE_uy∂x_b = spzeros(n_uy*n_b*2, n_b*4)
        
        for i in 1:n_ux

            Dux_x_i = Dux_x[:, i]
            Dux_y_i = Dux_y[:, i]
        
            Duy_x_i = Duy_x[:, i]
            Duy_y_i = Duy_y[:, i]
        
            ∂Dux∂rx_b_i = ∂Dux∂rx_b[:, i]
            ∂Dux∂ry_b_i = ∂Dux∂ry_b[:, i]
        
            ∂Duy∂rx_b_i = ∂Duy∂rx_b[:, i]
            ∂Duy∂ry_b_i = ∂Duy∂ry_b[:, i]

            ∂vecE_ux∂rx_b_i = ∂Dux∂rx_b_i .* Dux_y_i
            ∂vecE_ux∂ry_b_i = Dux_x_i .* ∂Dux∂ry_b_i
        
            ∂vecE_uy∂rx_b_i = ∂Duy∂rx_b_i .* Duy_y_i
            ∂vecE_uy∂ry_b_i = Duy_x_i .* ∂Duy∂ry_b_i

            ∂vecE_ux∂x_b[1 + n_b*2*(i-1):n_b*2*(i-1) + n_b, 1:end÷2] = hcat(spdiagm(∂vecE_ux∂rx_b_i), spdiagm(∂vecE_ux∂ry_b_i))
            ∂vecE_uy∂x_b[1 + n_b + n_b*2*(i-1):2*n_b + n_b*2*(i-1), 1:end÷2] = hcat(spdiagm(∂vecE_uy∂rx_b_i), spdiagm(∂vecE_uy∂ry_b_i))
        
        end

        ∂vecE∂x_b = vcat(∂vecE_ux∂x_b, ∂vecE_uy∂x_b)

    end

    if n_b >= 128

        I_ux_∂E_T = SharedVector{Int64}(n_ux*2*n_b)
        J_ux_∂E_T = SharedVector{Int64}(n_ux*2*n_b)
        values_ux_∂E_T = SharedVector{Float64}(n_ux*2*n_b)

        I_uy_∂E_T = SharedVector{Int64}(n_uy*2*n_b)
        J_uy_∂E_T = SharedVector{Int64}(n_ux*2*n_b)
        values_uy_∂E_T = SharedVector{Float64}(n_uy*2*n_b)

        @threads for i in 1:boundary.nodes

            Dux_x_i = Dux_x[i, :]
            Dux_y_i = Dux_y[i, :]

            Duy_x_i = Duy_x[i, :]
            Duy_y_i = Duy_y[i, :]

            ∂Dux∂rx_b_i = ∂Dux∂rx_b[i, :]
            ∂Dux∂ry_b_i = ∂Dux∂ry_b[i, :]

            ∂Duy∂rx_b_i = ∂Duy∂rx_b[i, :]
            ∂Duy∂ry_b_i = ∂Duy∂ry_b[i, :]

            if ~iszero(Dux_x_i) || ~iszero(Dux_y_i)

                ∂vecE_ux∂rx_b_i = ∂Dux∂rx_b_i .* Dux_y_i
                ∂vecE_ux∂ry_b_i = Dux_x_i .* ∂Dux∂ry_b_i

                I_ux_∂E_T[1 + (i-1)*2*n_ux:2*n_ux*i] = kron([1, 1], 1 + n_ux*2*(i-1):n_ux*(2*(i-1)+1))
                J_ux_∂E_T[1 + (i-1)*2*n_ux:2*n_ux*i] = vcat(i.*ones(n_ux), (i+n_b).*ones(n_ux))
                values_ux_∂E_T[1 + (i-1)*2*n_ux:2*n_ux*i] = vcat(∂vecE_ux∂rx_b_i, ∂vecE_ux∂ry_b_i)

            end

            if ~iszero(Duy_x_i) || ~iszero(Duy_y_i)
            
                ∂vecE_uy∂rx_b_i = ∂Duy∂rx_b_i .* Duy_y_i
                ∂vecE_uy∂ry_b_i = Duy_x_i .* ∂Duy∂ry_b_i
                
                I_uy_∂E_T[1 + (i-1)*2*n_uy:2*n_uy*i] = kron([1, 1], 1 + n_uy*(2*(i-1)+1):n_uy*2*(i))
                J_uy_∂E_T[1 + (i-1)*2*n_uy:2*n_uy*i] = vcat(i.*ones(n_uy), (i+n_b).*ones(n_uy))
                values_uy_∂E_T[1 + (i-1)*2*n_uy:2*n_uy*i] = vcat(∂vecE_uy∂rx_b_i, ∂vecE_uy∂ry_b_i)
            
            end

        end

        nonzeros_ux_∂E_T = findall(!iszero, vec(values_ux_∂E_T))
        nonzeros_uy_∂E_T = findall(!iszero, vec(values_uy_∂E_T))

        ∂vecE_T_ux∂x_b = sparse(vec(I_ux_∂E_T)[nonzeros_ux_∂E_T], vec(J_ux_∂E_T)[nonzeros_ux_∂E_T], vec(values_ux_∂E_T)[nonzeros_ux_∂E_T], n_ux*n_b*2, n_b*4)
        ∂vecE_T_uy∂x_b = sparse(vec(I_uy_∂E_T)[nonzeros_uy_∂E_T], vec(J_uy_∂E_T)[nonzeros_uy_∂E_T], vec(values_uy_∂E_T)[nonzeros_uy_∂E_T], n_uy*n_b*2, n_b*4)
        ∂vecE_T∂x_b = vcat(∂vecE_T_ux∂x_b, ∂vecE_T_uy∂x_b)

    else

        ∂vecE_T_ux∂x_b = spzeros(n_ux*n_b*2, n_b*4)
        ∂vecE_T_uy∂x_b = spzeros(n_uy*n_b*2, n_b*4)
        
        for i in 1:boundary.nodes

            Dux_x_i = Dux_x[i, :]
            Dux_y_i = Dux_y[i, :]
    
            Duy_x_i = Duy_x[i, :]
            Duy_y_i = Duy_y[i, :]
    
            ∂Dux∂rx_b_i = ∂Dux∂rx_b[i, :]
            ∂Dux∂ry_b_i = ∂Dux∂ry_b[i, :]
    
            ∂Duy∂rx_b_i = ∂Duy∂rx_b[i, :]
            ∂Duy∂ry_b_i = ∂Duy∂ry_b[i, :]
    
            ∂vecE_ux∂rx_b_i = ∂Dux∂rx_b_i .* Dux_y_i
            ∂vecE_ux∂ry_b_i = Dux_x_i .* ∂Dux∂ry_b_i
    
            ∂vecE_uy∂rx_b_i = ∂Duy∂rx_b_i .* Duy_y_i
            ∂vecE_uy∂ry_b_i = Duy_x_i .* ∂Duy∂ry_b_i
    
            ∂vecE_T_ux∂x_b[1 + n_ux*2*(i-1):n_ux*(2*(i-1)+1), i] = ∂vecE_ux∂rx_b_i
            ∂vecE_T_ux∂x_b[1 + n_ux*2*(i-1):n_ux*(2*(i-1)+1), i+n_b] = ∂vecE_ux∂ry_b_i
    
            ∂vecE_T_uy∂x_b[1 + n_uy*(2*(i-1)+1):n_uy*2*(i), i] = ∂vecE_uy∂rx_b_i
            ∂vecE_T_uy∂x_b[1 + n_uy*(2*(i-1)+1):n_uy*2*(i), i+n_b] = ∂vecE_uy∂ry_b_i
    
        end
    
        ∂vecE_T∂x_b = vcat(∂vecE_T_ux∂x_b, ∂vecE_T_uy∂x_b)
    end

    # calculate all kernel values corresponding to boundary nodes
    E_u = Dux_x .* Dux_y
    E_v = Duy_x .* Duy_y
    
    E = blockdiag(E_u, E_v)

    return E, ∂vecE∂x_b, ∂vecE_T∂x_b

end

function discrete_delta(r::AbstractMatrix)
    
    # construct initial d matrix
    m, n = size(r)
    
    # define logical arrays
    u05_i = getindex.(findall(x -> x <= 0.5, abs.(r)), 1)
    u05_j = getindex.(findall(x -> x <= 0.5, abs.(r)), 2)

    o05u15_i = getindex.(findall(x -> x > 0.5 && x <= 1.5, abs.(r)), 1)
    o05u15_j = getindex.(findall(x -> x > 0.5 && x <= 1.5, abs.(r)), 2)
    
    r_u05 = abs.(r[abs.(r) .<= 0.5])
    r_o05u15 = abs.(r[(abs.(r) .> 0.5) .* (abs.(r) .<= 1.5)])
    
    # calculate delta function values
    d_u05 = (1/3) .* (1 .+ sqrt.(-3 .* (r_u05.^2) .+ 1))
    d_o05u15 = (1/6) .* (5 .- 3 .* r_o05u15 .- sqrt.(-3 .* (1 .- r_o05u15).^2 .+ 1))
    
    d = sparse(vcat(u05_i, o05u15_i), vcat(u05_j, o05u15_j), vcat(d_u05, d_o05u15), m, n)

    return d
end
function discrete_delta(r::AbstractVector)
    
    # construct initial d matrix
    m = length(r)
    
    # define logical arrays
    u05_i = getindex.(findall(x -> x <= 0.5, abs.(r)))
    o05u15_i = getindex.(findall(x -> x > 0.5 && x <= 1.5, abs.(r)))
    
    r_u05 = abs.(r[abs.(r) .<= 0.5])
    r_o05u15 = abs.(r[(abs.(r) .> 0.5) .* (abs.(r) .<= 1.5)])
    
    # calculate delta function values
    d_u05 = (1/3) .* (1 .+ sqrt.(-3 .* (r_u05.^2) .+ 1))
    d_o05u15 = (1/6) .* (5 .- 3 .* r_o05u15 .- sqrt.(-3 .* (1 .- r_o05u15).^2 .+ 1))
    
    d = sparsevec(vcat(u05_i, o05u15_i), vcat(d_u05, d_o05u15), m)

    return d
end
function discrete_delta_jacobian(r::AbstractMatrix)
    
    # construct initial d matrix
    m, n = size(r)
    
    # define logical arrays
    u05_i = getindex.(findall(x -> x <= 0.5, abs.(r)), 1)
    u05_j = getindex.(findall(x -> x <= 0.5, abs.(r)), 2)

    o05u15_i = getindex.(findall(x -> x > 0.5 && x <= 1.5, abs.(r)), 1)
    o05u15_j = getindex.(findall(x -> x > 0.5 && x <= 1.5, abs.(r)), 2)
    
    r_u05 = r[abs.(r) .<= 0.5]
    r_o05u15 = r[(abs.(r) .> 0.5) .* (abs.(r) .<= 1.5)]
    
    # calculate delta function values
    d_u05_grad = -r_u05 ./ sqrt.(1 .- 3 .* (r_u05.^2))
    d_o05u15_grad = (sign.(r_o05u15)*(1/2)) .* (-1 .+ (-1 .+ abs.(r_o05u15)) ./ sqrt.(-2 .- 3 .* r_o05u15.^2 .+ 6 .* abs.(r_o05u15)))
    
    d_grad = sparse(vcat(u05_i, o05u15_i), vcat(u05_j, o05u15_j), vcat(d_u05_grad, d_o05u15_grad), m, n)

    return d_grad

end
function discrete_delta_jacobian(r::AbstractVector)
    
    # construct initial d matrix
    m = length(r)
    
    # define logical arrays
    u05_i = getindex.(findall(x -> x <= 0.5, abs.(r)))
    o05u15_i = getindex.(findall(x -> x > 0.5 && x <= 1.5, abs.(r)), 1)
    
    r_u05 = r[abs.(r) .<= 0.5]
    r_o05u15 = r[(abs.(r) .> 0.5) .* (abs.(r) .<= 1.5)]
    
    # calculate delta function values
    d_u05_grad = -r_u05 ./ sqrt.(1 .- 3 .* (r_u05.^2))
    d_o05u15_grad = (sign.(r_o05u15)*(1/2)) .* (-1 .+ (-1 .+ abs.(r_o05u15)) ./ sqrt.(-2 .- 3 .* r_o05u15.^2 .+ 6 .* abs.(r_o05u15)))
    
    d_grad = sparsevec(vcat(u05_i, o05u15_i), vcat(d_u05_grad, d_o05u15_grad), m)

    return d_grad

end

function meshgrid_vec(x_coord::AbstractVector, y_coord::AbstractVector)

    # make meshgrid
    x_grid = (x_coord)' .* ones(length(y_coord))
    y_grid = ones(length(x_coord))' .* y_coord

    x_coord_uyec = x_grid[:]
    y_yec = y_grid[:]

    return x_coord_uyec, y_yec
end