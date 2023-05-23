"""
2nd-order Finite-Volume discretization of fluid grid

NOTE: To follow traditional syntax in fluid literature,
instead of u = [ux, uy]', we define u = ux and v = uy in
this script.

"""
struct FVM_CDS_2D
    cv_avg::Vector{<:AbstractSparseArray} # determine avg control volume operators
    N::Vector{Vector{<:AbstractSparseArray}} # nonlinear convective operators
    L::Vector{<:AbstractSparseArray} # lagrangian operators
    D::Vector{<:AbstractSparseArray} # divergence operators
    G::SparseMatrixCSC{Float64, Int64} # gradient operator
end

function FVM_CDS_2D(ne::AbstractVector{Int64}=SA[10, 10],
    h::AbstractVector{<:AbstractFloat}=SA[1.0, 1.0],
    bc_vel::VecOrMat{<:AbstractVector}= [SA[0.0, 0.0], SA[0.0, 0.0], SA[0.0, 0.0], SA[0.0, 0.0]],
    outflow::AbstractVector{Bool}=SA[false, false, false, false])
    
    """
    This function creates the discrete operators needed to calculate the N-S
    equations using the projection method. This function specifically follows
    a finite volume discretization using the midpoint method and assumes a
    structured rectangular mesh in 2D.
    
    Inputs:

        ne = [ne_x, ne_y] vector containing number of control volumes in x and y directions
        h = [h_x, h_y] vector containing step size in x and y direction
        bc_vel = [bc_west, bc_east, bc_north, bc_south] array
                 containing velocity boundary conditions
        outflow = boolean array specifying which boundaries are outflows
    
    Outputs:
    
        FVM_CDS_2D = structure containing FVM operators
    """
    
    u_bind, v_bind, p_bind = boundary_ind(ne)

    # calculate operators
    N, N_bc = convective(ne, h, bc_vel, u_bind, v_bind, outflow)

    D, D_bc = divergence(ne, h, bc_vel, u_bind, p_bind, outflow)
    
    G = gradient(D)
    
    L, Lp, L_bc = lagrangian(ne, h, bc_vel, u_bind, v_bind, p_bind, outflow)
    
    cv_avg, cv_avg_bc = cv_average(ne, h, bc_vel, u_bind, v_bind, p_bind, outflow)
    
    return FVM_CDS_2D([cv_avg, cv_avg_bc], [N, N_bc], [L, Lp, L_bc], [D, D_bc], G)
end

function boundary_ind(ne::AbstractVector{Int64})

    """
    This function determines the indices of control volumes adjacent to boundaries

    Inputs:

        ne = [ne_x, ne_y] vector containing number of control volumes in x and y directions
    
    Outputs:
    
        u_bind, v_bind, p_bind = boundary indices for each control volume type
    """

    # extract edge densities
    ne_x, ne_y = ne
    
    # extract number of nodes for each N-S variable
    nf_u = (ne_x-1)*ne_y
    nf_v = ne_x*(ne_y-1)
    nf_p = ne_x*ne_y

    # Determine which indices of u, v, and p vectors correspond to boundary values
    west_u = collect(1 : ne_y)
    east_u = collect(nf_u-ne_y+1 : nf_u)
    north_u = collect(ne_y : ne_y : nf_u)
    south_u = collect(1 : ne_y : nf_u-ne_y+1)
    
    west_v = collect(1 : ne_y-1)
    east_v = collect(nf_v-ne_y+2 : nf_v)
    north_v = collect(ne_y-1 : ne_y-1 : nf_v)
    south_v = collect(1 : ne_y-1 : nf_v-ne_y+2)
    
    west_p = collect(1 : ne_y)
    east_p = collect(nf_p-ne_y+1 : nf_p)
    north_p = collect(ne_y : ne_y : nf_p)
    south_p = collect(1 : ne_y : nf_p-ne_y+1)
    
    u_bind = [west_u, east_u, north_u, south_u]
    v_bind = [west_v, east_v, north_v, south_v]
    p_bind = [west_p, east_p, north_p, south_p]

    return u_bind, v_bind, p_bind

end

function convective(ne::AbstractVector{Int64}, h::AbstractVector{<:AbstractFloat},
    bc_vel::VecOrMat{<:AbstractVector}, u_bind::VecOrMat{<:AbstractVector},
    v_bind::VecOrMat{<:AbstractVector}, outflow::AbstractVector{Bool})

    """
    This function calculates the midpoint operators used to determine the convective term
    corresponding to CDS of 2nd order FVM

    Inputs:

        ne = [ne_x, ne_y] vector containing number of control volumes in x and y directions
        h = [h_x, h_y] vector containing step size in x and y direction
        bc_vel = [bc_west, bc_east, bc_north, bc_south] array
                 containing velocity boundary conditions
        u_bind = boundary indices for u velocity cv
        v_bind = boundary indices for v velocity cv
        outflow = boolean array specifying which boundaries are outflows
    
    Outputs:
    
        vectors of midpoint operators and respective boundary terms
    """

    # extract edge densities
    ne_x, ne_y = ne
    
    # extract number of nodes for each N-S variable
    nf_u = (ne_x-1)*ne_y
    nf_v = ne_x*(ne_y-1)

    # extract step sizes
    h_x, h_y = h

    # extract velocities
    wb_vel, eb_vel, nb_vel, sb_vel = bc_vel
    
    # extract boundaries
    west_u, east_u, north_u, south_u = u_bind
    west_v, east_v, north_v, south_v = v_bind
    
    # extract outflow booleans
    outflow_w, outflow_e, outflow_n, outflow_s = outflow
    
    ## east u cv
    u_ue = (0.5/sqrt(h_x)).*spdiagm(0 => ones(nf_u), ne_y => ones(nf_u-ne_y))
    u_ue_bc = spzeros(nf_u)

    ## east v cv
    v_ve = (0.5/sqrt(h_x)).*spdiagm(0 => ones(nf_v), ne_y-1 => ones(nf_v-(ne_y-1)))
    v_ve[east_v, :] .= 0
    dropzeros!(v_ve)
    v_ve_bc = spzeros(nf_v)

    v_ue_uw_block1 = (0.5/sqrt(h_x)).*spdiagm(ne_y-1, ne_y, 0 => ones(ne_y-1), 1 => ones(ne_y-1))
    v_ue = kron(I(ne_x-1), v_ue_uw_block1)
    v_ue = vcat(v_ue, spzeros(ne_y-1, size(v_ue, 2)))
    v_ue_bc = spzeros(nf_v)

    ## east outflow bc
    if outflow_e
        u_ue[east_u, east_u] = (1/sqrt(h_x)).*spdiagm(ones(length(east_u)))
        v_ve[east_v, east_v] = (1/sqrt(h_x)).*spdiagm(ones(length(east_v)))
        v_ue[east_v, east_u] = v_ue_uw_block1
    else
        u_ue_bc[east_u] .= (0.5/sqrt(h_x)).*eb_vel[1]
        v_ve_bc[east_v] .= 1/sqrt(h_x).*eb_vel[2]
        v_ue_bc[east_v] .= 1/sqrt(h_x).*eb_vel[1]
    end

    ## west u cv
    u_uw = (0.5/sqrt(h_x)).*spdiagm(0 => ones(nf_u), -ne_y => ones(nf_u-ne_y))
    u_uw_bc = spzeros(nf_u)

    ## west v cv
    v_vw = (0.5/sqrt(h_x)).*spdiagm(0 => ones(nf_v), ne_y-1 => ones(nf_v-(ne_y-1)))'
    v_vw[west_v, :] .= 0
    dropzeros!(v_vw)
    v_vw_bc = spzeros(nf_v)

    v_uw = kron(I(ne_x-1), v_ue_uw_block1)
    v_uw = vcat(spzeros(ne_y-1, size(v_uw, 2)), v_uw)
    v_uw_bc = spzeros(nf_v)

    ## west outflow bc
    if outflow_w
        u_uw[west_u, west_u] = (1/sqrt(h_x)).*spdiagm(ones(length(west_u)))
        v_vw[west_v, west_v] = (1/sqrt(h_x)).*spdiagm(ones(length(west_v)))
        v_uw[west_v, west_u] = v_ue_uw_block1
    else
        u_uw_bc[west_u] .= (0.5/sqrt(h_x)).*wb_vel[1]
        v_vw_bc[west_v] .= (1/sqrt(h_x)).*wb_vel[2]
        v_uw_bc[west_v] .= (1/sqrt(h_x)).*wb_vel[1]
    end

    ## north u cv
    u_un = (0.5/sqrt(h_y)).*spdiagm(0 => ones(nf_u), 1 => ones(nf_u-1))
    u_un[north_u, :] .= 0
    dropzeros!(u_un)
    u_un_bc = spzeros(nf_u)

    u_vn_block1 = (0.5/sqrt(h_y)).*spdiagm(ne_y, ne_y-1, 0 => ones(ne_y-1))
    u_vn_block2 = spdiagm(ne_x-1, ne_x, 0 => ones(ne_x-1), 1 => ones(ne_x-1))
    u_vn = kron(u_vn_block2, u_vn_block1)
    u_vn_bc = spzeros(nf_u)

    ## north v cv
    v_vn_block1 = (0.5/sqrt(h_y)).*spdiagm(0 => ones(ne_y-1), 1 => ones(ne_y-2))
    v_vn = kron(I(ne_x), v_vn_block1)
    v_vn_bc = spzeros(nf_v)

    ## north outflow bc
    if outflow_n
        u_un[north_u, north_u] = (1/sqrt(h_y)).*spdiagm(ones(length(north_u)))
        u_vn[north_u, :] = u_vn[north_u.-1, :]
        v_vn[north_v, north_v] = (1/sqrt(h_y)).*spdiagm(ones(length(north_v)))
    else
        u_un_bc[north_u] .= (1/sqrt(h_y)).*nb_vel[1]
        u_vn_bc[north_u] .= (1/sqrt(h_y)).*nb_vel[2]
        v_vn_bc[north_v] .= (0.5/sqrt(h_y)).*nb_vel[2]
    end

    ## south u cv
    u_us = (0.5/sqrt(h_y)).*spdiagm(0 => ones(nf_u), 1 => ones(nf_u-1))'
    u_us[south_u, :] .= 0
    dropzeros!(u_us)
    u_us_bc = spzeros(nf_u)

    u_vs_block1 = (0.5/sqrt(h_y)).*spdiagm(ne_y, ne_y-1, -1 => ones(ne_y-1))
    u_vs_block2 = spdiagm(ne_x-1, ne_x, 0 => ones(ne_x-1), 1 => ones(ne_x-1))
    u_vs = kron(u_vs_block2, u_vs_block1)
    u_vs_bc = spzeros(nf_u)

    ## south v cv
    v_vs_block1 = (0.5/sqrt(h_y)).*spdiagm(0 => ones(ne_y-1), 1 => ones(ne_y-2))'
    v_vs = kron(I(ne_x), v_vs_block1)
    v_vs_bc = spzeros(nf_v)

    if outflow_s
        u_us[south_u, south_u] = (1/sqrt(h_y)).*spdiagm(ones(length(south_u)))
        u_vs[south_u, :] = u_vs[south_u.+1, :]
        v_vs[south_v, south_v] = (1/sqrt(h_y)).*spdiagm(ones(length(south_v)))
    else
        u_us_bc[south_u] .= (1/sqrt(h_y)).*sb_vel[1]
        u_vs_bc[south_u] .= (1/sqrt(h_y)).*sb_vel[2]
        v_vs_bc[south_v] .= (0.5/sqrt(h_y)).*sb_vel[2]
    end

    # combine directional midpoint operators
    m1 = cat(u_ue, v_vn; dims=(1,2))
    m1_bc = vcat(u_ue_bc, v_vn_bc)

    m2 = cat(u_uw, v_vs; dims=(1,2))
    m2_bc = vcat(u_uw_bc, v_vs_bc)

    m3 = cat(u_un, v_ve; dims=(1,2))
    m3_bc = vcat(u_un_bc, v_ve_bc)

    m4 = vcat(hcat(spzeros(nf_u, nf_u), u_vn),
        hcat(v_ue, spzeros(nf_v, nf_v)))
    m4_bc = vcat(u_vn_bc, v_ue_bc)

    m5 = cat(u_us, v_vw; dims=(1,2))
    m5_bc = vcat(u_us_bc, v_vw_bc)

    m6 = vcat(hcat(spzeros(nf_u, nf_u), u_vs),
        hcat(v_uw, spzeros(nf_v, nf_v)))
    m6_bc = vcat(u_vs_bc, v_uw_bc)

    return [m1, m2, m3, m4, m5, m6], [m1_bc, m2_bc, m3_bc, m4_bc, m5_bc, m6_bc]
end

function divergence(ne::AbstractVector{Int64}, h::AbstractVector{<:AbstractFloat},
    bc_vel::VecOrMat{<:AbstractVector}, u_bind::VecOrMat{<:AbstractVector},
    p_bind::VecOrMat{<:AbstractVector}, outflow::AbstractVector{Bool})

    """
    This function determines the divergence operator corresponding to CDS of 2nd order FVM

    Inputs:

        ne = [ne_x, ne_y] vector containing number of control volumes in x and y directions
        h = [h_x, h_y] vector containing step size in x and y direction
        bc_vel = [bc_west, bc_east, bc_north, bc_south] array
                 containing velocity boundary conditions
        u_bind = boundary indices for u velocity cv
        p_bind = boundary indices for pressure cv
        outflow = boolean array specifying which boundaries are outflows
    
    Outputs:
    
        D = divergence operator (a sparse matrix)
        D_bc = boundary condition term
    """

    # extract edge densities
    ne_x, ne_y = ne
    
    # calculate number of nodes for each N-S variable
    nf_u = (ne_x-1)*ne_y
    nf_v = ne_x*(ne_y-1)
    nf_p = ne_x*ne_y

    # extract step sizes
    h_x, h_y = h

    # extract velocities
    wb_vel, eb_vel, nb_vel, sb_vel = bc_vel
    
    # extract boundaries
    west_u, east_u, _, _ = u_bind
    west_p, east_p, north_p, south_p = p_bind
    
    # extract outflow booleans
    outflow_w, outflow_e, outflow_n, outflow_s = outflow

    ## divergence u
    Du = (1/h_x).*spdiagm(nf_p, nf_u, 0 => ones(nf_u), -ne_y => -ones(nf_u))
    Du_bc = spzeros(nf_p)

    ## divergence v
    Dv_block1 = (1/h_y).*spdiagm(ne_y, ne_y-1, 0 => ones(ne_y-1), -1 => -ones(ne_y-1))
    Dv = kron(I(ne_x), Dv_block1)
    Dv_bc = spzeros(nf_p)

    ## outflow bc
    if outflow_e
        Du[east_p, east_u] .= 0
    else
        Du_bc[east_p] .= (1/h_x).*eb_vel[1]
    end

    if outflow_w
        Du[west_p, west_u] .= 0
    else
        Du_bc[west_p] .= -(1/h_x).*wb_vel[1]
    end

    if outflow_n
        Dv[north_p, :] .= 0
    else
        Dv_bc[north_p] .= (1/h_y).*nb_vel[2]
    end

    if outflow_s
        Dv[south_p, :] .= 0
    else
        Dv_bc[south_p] .= -(1/h_y).*sb_vel[2]
    end

    dropzeros!(Du)
    dropzeros!(Dv)

    D = hcat(Du, Dv)
    D_bc = Du_bc + Dv_bc

    return D, D_bc
end

function gradient(D::AbstractMatrix{<:AbstractFloat})
    
    """
    This function determines the gradient operator corresponding to CDS of 2nd order FVM
    as the negative inverse of the divergence

    Inputs:

        D = gradient operator (a sparse matrix)
    
    Outputs:
    
        G = gradient operator (a sparse matrix)
    """
    
    G = sparse(-D')

    return G
end

function lagrangian(ne::AbstractVector{Int64}, h::AbstractVector{<:AbstractFloat},
    bc_vel::VecOrMat{<:AbstractVector}, u_bind::VecOrMat{<:AbstractVector},
    v_bind::VecOrMat{<:AbstractVector}, p_bind::VecOrMat{<:AbstractVector},
    outflow::AbstractVector{Bool})

    """
    This function calculates the lagrangian operator corresponding to CDS of 2nd order FVM

    Inputs:

        ne = [ne_x, ne_y] vector containing number of control volumes in x and y directions
        h = [h_x, h_y] vector containing step size in x and y direction
        bc_vel = [bc_west, bc_east, bc_north, bc_south] array
                 containing velocity boundary conditions
        u_bind = boundary indices for u velocity cv
        v_bind = boundary indices for v velocity cv
        v_bind = boundary indices for pressure cv
        outflow = boolean array specifying which boundaries are outflows
    
    Outputs:
    
        L = lagrangian operator (a sparse matrix)
        Lp = lagrangian operator for pressure cv (a sparse matrix)
        L_bc = boundary condition term
    """

    # extract edge densities
    ne_x, ne_y = ne
    
    # calculate number of nodes for each N-S variable
    nf_u = (ne_x-1)*ne_y
    nf_v = ne_x*(ne_y-1)
    nf_p = ne_x*ne_y

    # extract step sizes
    h_x, h_y = h

    # extract velocities
    wb_vel, eb_vel, nb_vel, sb_vel = bc_vel
    
    # extract boundaries
    west_u, east_u, north_u, south_u = u_bind
    west_v, east_v, north_v, south_v = v_bind
    west_p, east_p, north_p, south_p = p_bind
    
    # extract outflow booleans
    outflow_w, outflow_e, outflow_n, outflow_s = outflow

    ## lagrangian u cv x direction
    Lu_x = (1/(h_x^2)).*spdiagm(0 => -2 .* ones(nf_u), ne_y => ones(nf_u-ne_y),
        -ne_y => ones(nf_u-ne_y))
    Lu_x_bc = spzeros(nf_u)
    
    ## lagrangian u cv y direction
    Lu_y_block1 = (1/(h_y^2)).*spdiagm(0 => -2 .* ones(ne_y), 1 => ones(ne_y-1),
        -1 => ones(ne_y-1))
    Lu_y_block1[1, 1] = -3/(h_y^2)
    Lu_y_block1[end, end] = -3/(h_y^2)
    Lu_y = kron(I(ne_x-1), Lu_y_block1)
    Lu_y_bc = spzeros(nf_u)

    ## lagrangian v cv x direction
    Lv_x = (1/(h_x^2)).*spdiagm(0 => -2 .* ones(nf_v), ne_y-1 => ones(nf_v-(ne_y-1)),
        -(ne_y-1) => ones(nf_v-(ne_y-1)))
    Lv_x[east_v, east_v] = -3/(h_y^2).*I(length(east_v))
    Lv_x[west_v, west_v] = -3/(h_y^2).*I(length(west_v))
    Lv_x_bc = spzeros(nf_v)
    
    ## lagrangian v cv y direction
    Lv_y_block1 = (1/(h_y^2)).*spdiagm(0 => -2 .* ones(ne_y-1), 1 => ones(ne_y-2),
        -1 => ones(ne_y-2))
    Lv_y = kron(I(ne_x), Lv_y_block1)
    Lv_y_bc = spzeros(nf_v)

    ## lagrangian p cv x direction
    Lp_x = (1/(h_x^2)).*spdiagm(0 => -2 .* ones(nf_p), ne_y => ones(nf_p-ne_y), -ne_y => ones(nf_p-ne_y))
    Lp_x[east_p, east_p] = -(1/(h_x^2)).*I(length(east_p))
    Lp_x[west_p, west_p] = -(1/(h_x^2)).*I(length(west_p))

    ## lagrangian p cv y direction
    Lp_y_block1 = (1/(h_y^2)).*spdiagm(0 => -2 .* ones(ne_y), 1 => ones(ne_y-1), -1 => ones(ne_y-1))
    Lp_y_block1[1, 1] = -1/(h_y^2)
    Lp_y_block1[end, end] = -1/(h_y^2)
    Lp_y = kron(I(ne_x), Lp_y_block1)

    ## outflow bc
    if outflow_e
        Lu_x[east_u, east_u] = (1/(h_x^2)).*spdiagm(-ones(length(east_u)))
        Lv_x[east_v, east_v] = (1/(h_x^2)).*spdiagm(-ones(length(east_v)))
    else
        Lu_x_bc[east_u] .= (1/(h_x^2)).*eb_vel[1]
        Lv_x_bc[east_v] .= (2/(h_x^2)).*eb_vel[2]
    end
    
    if outflow_w
        Lu_x[west_u, west_u] = (1/(h_x^2)).*spdiagm(-ones(length(west_u)))
        Lv_x[west_v, west_v] = (1/(h_x^2)).*spdiagm(-ones(length(west_v)))
    else
        Lu_x_bc[west_u] .= (1/(h_x^2)).*wb_vel[1]
        Lv_x_bc[west_v] .= (2/(h_x^2)).*wb_vel[2]
    end

    if outflow_n
        Lu_y[north_u, north_u] = (1/(h_y^2)).*spdiagm(-ones(length(north_u)))
        Lv_y[north_v, north_v] = (1/(h_y^2)).*spdiagm(-ones(length(north_v)))
    else
        Lu_y_bc[north_u] .= (2/(h_y^2)).*nb_vel[1]
        Lv_y_bc[north_v] .= (1/(h_y^2)).*nb_vel[2]
    end
    
    if outflow_s
        Lu_y[south_u, south_u] .= (1/(h_y^2)).*spdiagm(-ones(length(south_u)))
        Lv_y[south_v, south_v] .= (1/(h_y^2)).*spdiagm(-ones(length(south_v)))
    else
        Lu_y_bc[south_u] .= (2/(h_y^2)).*sb_vel[1]
        Lv_y_bc[south_v] .= (1/(h_y^2)).*sb_vel[2]
    end

    ## combine
    Lp = Lp_x + Lp_y

    Lu = Lu_x + Lu_y
    Lu_bc = Lu_x_bc + Lu_y_bc

    Lv = Lv_x + Lv_y
    Lv_bc = Lv_x_bc + Lv_y_bc

    L = cat(Lu, Lv; dims=(1,2))
    L_bc = vcat(Lu_bc, Lv_bc)

    return L, Lp, L_bc
end

function cv_average(ne::AbstractVector{Int64}, h::AbstractVector{<:AbstractFloat},
    bc_vel::VecOrMat{<:AbstractVector}, u_bind::VecOrMat{<:AbstractVector},
    v_bind::VecOrMat{<:AbstractVector}, p_bind::VecOrMat{<:AbstractVector},
    outflow::AbstractVector{Bool})

    """
    This function determines the operators used to average velocities over pressure
    cv corresponding to CDS of 2nd order FVM

    Inputs:

        ne = [ne_x, ne_y] vector containing number of control volumes in x and y directions
        h = [h_x, h_y] vector containing step size in x and y direction
        bc_vel = [bc_west, bc_east, bc_north, bc_south] array
                 containing velocity boundary conditions
        u_bind = boundary indices for u velocity cv
        p_bind = boundary indices for pressure cv
        outflow = boolean array specifying which boundaries are outflows
    
    Outputs:
    
        cv_avg = averaging operator (a sparse matrix)
        cv_avg_bc = boundary condition term
    """

    # extract edge densities
    ne_x, ne_y = ne
    
    # calculate number of nodes for each N-S variable
    nf_u = (ne_x-1)*ne_y
    nf_v = ne_x*(ne_y-1)
    nf_p = ne_x*ne_y

    # extract step sizes
    h_x, h_y = h

    # extract velocities
    wb_vel, eb_vel, nb_vel, sb_vel = bc_vel
    
    # extract boundaries
    west_u, east_u, _, _ = u_bind
    west_p, east_p, north_p, south_p = p_bind
    _, _, north_v, south_v = v_bind
    
    # extract outflow booleans
    outflow_w, outflow_e, outflow_n, outflow_s = outflow

    ## p cv average u
    cv_avg_u = 0.5 .* spdiagm(nf_p, nf_u, 0 => ones(nf_u), -ne_y => ones(nf_u))
    cv_avg_u_bc = spzeros(nf_p)

    ## p cv average v
    cv_avg_v_block1 = 0.5 .* spdiagm(ne_y, ne_y-1, 0 => ones(ne_y-1), -1 => ones(ne_y-1))
    cv_avg_v = kron(I(ne_x), cv_avg_v_block1)
    cv_avg_v_bc = spzeros(nf_p)

    ## outflow bc
    if outflow_e
        cv_avg_u[east_p, east_u] .= spdiagm(ones(length(east_p)))
    else
        cv_avg_u_bc[east_p] .= 0.5 .* eb_vel[1]
    end

    if outflow_w
        cv_avg_u[west_p, west_u] .= spdiagm(ones(length(west_p)))
    else
        cv_avg_u_bc[west_p] .= 0.5 .* wb_vel[1]
    end

    if outflow_n
        cv_avg_v[north_p, north_v] .= spdiagm(ones(length(north_p)))
    else
        cv_avg_v_bc[north_p] .= 0.5 .* nb_vel[2]
    end

    if outflow_s
        cv_avg_v[south_p, south_v] .= spdiagm(ones(length(south_p)))
    else
        cv_avg_v_bc[south_p] .= 0.5 .* sb_vel[2]
    end

    dropzeros!(cv_avg_u)
    dropzeros!(cv_avg_v)

    cv_avg = cat(cv_avg_u, cv_avg_v; dims=(1,2))
    cv_avg_bc = vcat(cv_avg_u_bc, cv_avg_v_bc)

    return cv_avg, cv_avg_bc
end