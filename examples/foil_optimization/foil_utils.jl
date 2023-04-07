function normalize_θ(model::DiamondFoil, θ::AbstractVector, ref_L::AbstractFloat, ref_u::AbstractFloat)
    
    return [θ[1], θ[2]/ref_L, θ[3], θ[4]]

end
function unnormalize_θ(model::DiamondFoil, θ::AbstractVector, ref_L::AbstractFloat, ref_u::AbstractFloat)

    return [θ[1], θ[2]*ref_L, θ[3], θ[4]]

end

function normalize_θg(model::DiamondFoil, θg::AbstractVector, ref_L::AbstractFloat, ref_u::AbstractFloat)
    
    return [θg[1]/ref_L, θg[2], θg[3]]

end
function unnormalize_θg(model::DiamondFoil, θg::AbstractVector, ref_L::AbstractFloat, ref_u::AbstractFloat)

    return [θg[1]*ref_L, θg[2], θg[3]]

end

function normalize_θs(model::DiamondFoil, θs::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)

    return θs

end
function unnormalize_θs(model::DiamondFoil, θs::AbstractVector, ref_L::AbstractFloat, ref_U::AbstractFloat)

    return θs

end

function oscillating_motion_xb(model::DiamondFoil, x0, θg, α, T)

    A = θg[1]
    B = θg[2]
    ϕ_pitch = θg[3]

    heave = A.*sin.(2*pi*α.*T) .+ x0[2]
    pitch = B.*sin.(2*pi*α.*T .- ϕ_pitch) .+ x0[3]

    vy = (A*2*pi*α).*cos.(2*pi*α.*T)
    ω = (B*2*pi*α).*cos.(2*pi*α.*T .- ϕ_pitch)

    X_b = [boundary_state(model, [x0[1], heave[i], pitch[i], 0, vy[i], ω[i]]) for i in eachindex(T)]

    return X_b

end
function oscillating_motion_x(model::DiamondFoil, x0, θg, α, T)

    A = θg[1]
    B = θg[2]
    ϕ_pitch = θg[3]

    heave = A.*sin.(2*pi*α.*T) .+ x0[2]
    pitch = B.*sin.(2*pi*α.*T .- ϕ_pitch) .+ x0[3]

    vy = (A*2*pi*α).*cos.(2*pi*α.*T)
    ω = (B*2*pi*α).*cos.(2*pi*α.*T .- ϕ_pitch)

    X = [[x0[1], heave[i], pitch[i], 0, vy[i], ω[i]] for i in eachindex(T)]

    return X

end

function x_θg_jacobian(model::DiamondFoil, θg::AbstractVector, t::AbstractFloat; α=1.0)

    B = θg[2]
    ϕ_pitch = θg[3]

    ∂x∂A = [0, sin(2*pi*α*t), 0, 0, (2*pi*α)*cos(2*pi*α*t), 0]
    ∂x∂B = [0, 0, sin(2*pi*α*t - ϕ_pitch), 0, 0, (2*pi*α)*cos(2*pi*α*t - ϕ_pitch)]

    ∂x∂ϕ_pitch = [0, 0, -B*cos(2*pi*α*t - ϕ_pitch), 0, 0, (B*2*pi*α)*sin(2*pi*α*t - ϕ_pitch)]

    return sparse(hcat(∂x∂A, ∂x∂B, ∂x∂ϕ_pitch))

end
function x_b_θs_jacobian(model::DiamondFoil, θs::AbstractVector, x::AbstractVector)

    leading_ratio = model.leading_ratio
    nodes = model.nodes
    c = model.cl
    θs = θs[1]

    nodes_e_vec = round.(Int, nodes.*[leading_ratio / (2 + 2*leading_ratio), 1 / (2 + 2*leading_ratio),
        1 / (2 + 2*leading_ratio), leading_ratio / (2 + 2*leading_ratio)]
    )

    θ2 = asin((1/leading_ratio)*sin(θs))
    dθ2dθs = ((1/leading_ratio)*cos(θs))*(1 / sqrt(1-((1/leading_ratio)*sin(θs))^2))

    # determine x and y coordinates of lagrangian points
    dvertices_xdθs = [c*leading_ratio*sin(θ2)*dθ2dθs, 0, -c*sin(θs), 0]
    dvertices_ydθs = [0, c*cos(θs), 0, -c*cos(θs)]
 
    drx0_bdθs = zeros(nodes)
    dry0_bdθs = zeros(nodes)

    for i in 0:2

        nodes_e = nodes_e_vec[i+1]
        ind_l = sum(nodes_e_vec[1:i]) + 1
        ind_b = sum(nodes_e_vec[1:i]) + nodes_e

        drx0_bdθs[ind_l:ind_b] = LinRange(dvertices_xdθs[i+1], dvertices_xdθs[i+2], nodes_e+1)[1:end-1]
        dry0_bdθs[ind_l:ind_b] = LinRange(dvertices_ydθs[i+1], dvertices_ydθs[i+2], nodes_e+1)[1:end-1]

    end

    nodes_e = nodes_e_vec[4]
    ind_l = sum(nodes_e_vec[1:3]) + 1
    ind_b = sum(nodes_e_vec[1:3]) + nodes_e

    drx0_bdθs[ind_l:ind_b] = LinRange(dvertices_xdθs[4], dvertices_xdθs[1], nodes_e+1)[1:end-1]
    dry0_bdθs[ind_l:ind_b] = LinRange(dvertices_ydθs[4], dvertices_ydθs[1], nodes_e+1)[1:end-1]

    Q = Rotations.RotMatrix(x[3])
    R = model.R
    Q_b = R'*kron(I(model.nodes), Q)*R

    dr_bdθs = Q_b*vcat(drx0_bdθs, dry0_bdθs)
    du_bdθs = Q_b*(x[6] .* vcat(-dry0_bdθs, drx0_bdθs))

    return vcat(dr_bdθs, du_bdθs)

end