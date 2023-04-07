import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Aquarium
using LinearAlgebra
using SparseArrays
using StaticArrays
using Makie
using Makie.GeometryBasics
using CairoMakie
using Pardiso
using JLD2
using Test
using ProgressMeter
using Interpolations
using ForwardDiff
using FiniteDiff
using BenchmarkTools
using Rotations

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)

##############################
## helper functions
##############################

function vec_E(model, boundary, x_b)
    vec_Et = vec(generate_E(model, boundary, x_b))
    return vec_Et
end

function vec_E_transpose(model, boundary, x_b)
    vec_Et = vec(generate_E(model, boundary, x_b)')
    return vec_Et
end

function boundary_coupling_x(model, boundary, x)
    x_b = Aquarium.boundary_state(boundary, x)
    E = boundary_coupling(model, boundary, x_b)

    return E
end

function full_fsi_residual(model::FSIModel, boundary::ImmersedBoundary,
    gk::AbstractVector, gkm1::AbstractVector)

    nf_u = (model.ne_x - 1) * model.ne_y + model.ne_x * (model.ne_y - 1)
    m_D = model.ne_x * model.ne_y
    m_E = 2 * boundary.nodes
    m_x = length(gk) - nf_u - m_D - m_E

    uk = gk[1:nf_u]
    pk = gk[nf_u+1:nf_u+m_D]
    fk_b = gk[nf_u+m_D+1:nf_u+m_D+m_E]
    xk = gk[end-m_x+1:end]

    ukm1 = gkm1[1:nf_u]

    r1_eval = Aquarium.R1(model, boundary, uk, pk, fk_b, xk, ukm1)
    c1_eval = Aquarium.c1(model, uk)
    c2_eval = Aquarium.c2(model, boundary, uk, xk)

    return vcat(r1_eval, c1_eval, c2_eval)
    
end

function dfdfk_b(fk_b)

    u1 = ones(length(fk_b)÷2)
    u2 = zeros(length(fk_b)÷2)

    u = vcat(u1, u2)

    ∂f∂fk_b = sparse(u'*sparse(I, length(fk_b), length(fk_b)))

    return ∂f∂fk_b

end

function sum_force(fk_b)

    fk_x_b = fk_b[1:end÷2]
    return sum(fk_x_b)

end

function f1(fluid::FSIModel, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θ_normalized;
    dt=0.001, t0=0.0, tf=0.01, λ1=1e-6, tol=1e-12, verbose=false)

    θs_normalized = [θ_normalized[1]]
    θg_normalized = θ_normalized[2:end]

    α = α_normalized / (fluid.ref_L / fluid.ref_u)
    θs = unnormalize_θs(DiamondFoil(;nodes=4), θs_normalized, fluid.ref_L, fluid.ref_u)
    θg = unnormalize_θg(DiamondFoil(;nodes=4), θg_normalized, fluid.ref_L, fluid.ref_u)

    boundary = Aquarium.DiamondFoil(ρ_b, θs[1], c; leading_ratio = leading_ratio, nodes=nodes)

    x = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], θg, α, t0:dt:tf)

    normalized_fluid = Aquarium.normalize(fluid)
    normalized_boundary = Aquarium.normalize(boundary, fluid.ref_L)

    x_normalized = map((x) -> Aquarium.normalize(boundary, x, fluid.ref_L, fluid.ref_u), x)
    
    uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
    t_hist, _, _, _, f_b_hist = simulate(normalized_fluid, normalized_boundary, 
        uk, pk, fk_b, x_normalized; λ1=λ1, tol=tol, t0=t0, tf=tf, alg=:pardiso,
        iter_refine=false, verbose=verbose)

    net_fx_hist = zeros(length(f_b_hist))

    for i in eachindex(f_b_hist)[2:end]
        fi_b = f_b_hist[i]
        fi_x_b = fi_b[1:normalized_boundary.nodes]
        net_fx_hist[i] = sum(fi_x_b)
    end

    period = 1/(2*α)
    valid_t = t_hist .>= t_hist[end] - period
    t_avg = dt / (t_hist[valid_t][end]-t_hist[valid_t][1])

    net_fx = t_avg*sum(net_fx_hist[valid_t])

    return net_fx

end

function f2(fluid::FSIModel, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θ_normalized;
    dt=0.001, t0=0.0, tf=0.01, λ1=1e-6, tol=1e-12, verbose=false)

    θs_normalized = [θ_normalized[1]]
    θg_normalized = θ_normalized[2:end]

    α = α_normalized / (fluid.ref_L / fluid.ref_u)
    θs = unnormalize_θs(DiamondFoil(;nodes=4), θs_normalized, fluid.ref_L, fluid.ref_u)
    θg = unnormalize_θg(DiamondFoil(;nodes=4), θg_normalized, fluid.ref_L, fluid.ref_u)

    boundary = Aquarium.DiamondFoil(ρ_b, θs[1], c; leading_ratio = leading_ratio, nodes=nodes)

    x = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], θg, α, t0:dt:tf)

    normalized_fluid = Aquarium.normalize(fluid)
    normalized_boundary = Aquarium.normalize(boundary, fluid.ref_L)

    x_normalized = map((x) -> Aquarium.normalize(boundary, x, fluid.ref_L, fluid.ref_u), x)
    
    uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
    _, _, u_hist, _, _ = simulate(normalized_fluid, normalized_boundary, 
        uk, pk, fk_b, x_normalized; λ1=λ1, tol=tol, t0=t0, tf=tf, alg=:pardiso,
        iter_refine=false, verbose=verbose)
    
    return u_hist[end]

end

function f3(fluid::FSIModel, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θ_normalized;
    dt=0.001, t0=0.0, tf=0.01, λ1=1e-6, tol=1e-12, verbose=false)

    θs_normalized = [θ_normalized[1]]
    θg_normalized = θ_normalized[2:end]

    α = α_normalized / (fluid.ref_L / fluid.ref_u)
    θs = unnormalize_θs(DiamondFoil(;nodes=4), θs_normalized, fluid.ref_L, fluid.ref_u)
    θg = unnormalize_θg(DiamondFoil(;nodes=4), θg_normalized, fluid.ref_L, fluid.ref_u)

    boundary = Aquarium.DiamondFoil(ρ_b, θs[1], c; leading_ratio = leading_ratio, nodes=nodes)

    x = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], θg, α, t0:dt:tf)

    normalized_fluid = Aquarium.normalize(fluid)
    normalized_boundary = Aquarium.normalize(boundary, fluid.ref_L)

    x_normalized = map((x) -> Aquarium.normalize(boundary, x, fluid.ref_L, fluid.ref_u), x)
    
    uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
    _, _, _, p_hist, _ = simulate(normalized_fluid, normalized_boundary, 
        uk, pk, fk_b, x_normalized; λ1=λ1, tol=tol, t0=t0, tf=tf, alg=:pardiso,
        iter_refine=false, verbose=verbose)
    
    return p_hist[end]

end

function f_shape(fluid::FSIModel, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θs_normalized, θg_normalized;
    dt=0.001, t0=0.0, tf=0.01, λ1=1e-6, tol=1e-12, verbose=false)

    α = α_normalized / (fluid.ref_L / fluid.ref_u)
    θs = unnormalize_θs(DiamondFoil(;nodes=4), θs_normalized, fluid.ref_L, fluid.ref_u)
    θg = unnormalize_θg(DiamondFoil(;nodes=4), θg_normalized, fluid.ref_L, fluid.ref_u)

    boundary = Aquarium.DiamondFoil(ρ_b, θs[1], c; leading_ratio = leading_ratio, nodes=nodes)

    x = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], θg, α, t0:dt:tf)

    normalized_fluid = Aquarium.normalize(fluid)
    normalized_boundary = Aquarium.normalize(boundary, fluid.ref_L)

    x_normalized = map((x) -> Aquarium.normalize(boundary, x, fluid.ref_L, fluid.ref_u), x)
    
    uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
    t_hist, _, _, _, f_b_hist, dudθ_hist, dpdθ_hist, df_bdθ_hist = simulate_diff(
        normalized_fluid, normalized_boundary, uk, pk, fk_b, x_normalized, θs_normalized, θg_normalized;
        λ1=λ1, tol=tol, t0=t0, tf=tf, alg=:pardiso, iter_refine=false, verbose=verbose, α=α_normalized
    )

    dudθs_hist = [dudθ_hist[i][:, 1:length(θs)] for i in eachindex(f_b_hist)]
    dpdθs_hist = [dpdθ_hist[i][:, 1:length(θs)] for i in eachindex(f_b_hist)]

    net_fx_hist = zeros(length(f_b_hist))
    dnext_fxdθs_hist = [zeros(1, length(θs)) for _ in eachindex(f_b_hist)]

    for i in eachindex(f_b_hist)[2:end]
        fi_b = f_b_hist[i]
        fi_x_b = fi_b[1:normalized_boundary.nodes]
        net_fx_hist[i] = sum(fi_x_b)
        dnext_fxdθs_hist[i] = dfdfk_b(fi_b)*df_bdθ_hist[i][:, 1:length(θs)]
    end

    period = 1/(2*α)
    valid_t = t_hist .>= t_hist[end] - period

    t_avg = dt / (t_hist[valid_t][end]-t_hist[valid_t][1])

    net_fx = t_avg*sum(net_fx_hist[valid_t])
    net_fxdθs = t_avg .* sum(dnext_fxdθs_hist[valid_t])

    save_file = joinpath(DATADIR, "foil_shape_optimization_BFGS_data_test.jld2")
    jldsave(save_file; dudθs_hist, dpdθs_hist, net_fxdθs)
    
    return net_fx

end
function g_shape(θ_normalized)

    data = load(joinpath(DATADIR, "foil_shape_optimization_BFGS_data_test.jld2"))
    dudθs_hist = data["dudθs_hist"]
    dpdθs_hist = data["dpdθs_hist"]
    net_fxdθs = data["net_fxdθs"]

    return dudθs_hist, dpdθs_hist, net_fxdθs

end

function f_gait(fluid::FSIModel, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θs_normalized, θg_normalized;
    dt=0.001, t0=0.0, tf=0.01, λ1=1e-6, tol=1e-12, verbose=false)

    α = α_normalized / (fluid.ref_L / fluid.ref_u)
    θs = unnormalize_θs(DiamondFoil(;nodes=4), θs_normalized, fluid.ref_L, fluid.ref_u)
    θg = unnormalize_θg(DiamondFoil(;nodes=4), θg_normalized, fluid.ref_L, fluid.ref_u)

    boundary = Aquarium.DiamondFoil(ρ_b, θs[1], c; leading_ratio = leading_ratio, nodes=nodes)

    x = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], θg, α, t0:dt:tf)

    normalized_fluid = Aquarium.normalize(fluid)
    normalized_boundary = Aquarium.normalize(boundary, fluid.ref_L)

    x_normalized = map((x) -> Aquarium.normalize(boundary, x, fluid.ref_L, fluid.ref_u), x)
    
    uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
    t_hist, _, _, _, f_b_hist, dudθ_hist, dpdθ_hist, df_bdθ_hist = simulate_diff(
        normalized_fluid, normalized_boundary, uk, pk, fk_b, x_normalized, θs_normalized, θg_normalized;
        λ1=λ1, tol=tol, t0=t0, tf=tf, alg=:pardiso, iter_refine=false, verbose=verbose, α=α_normalized
    )

    dudθg_hist = [dudθ_hist[i][:, length(θs)+1:end] for i in eachindex(f_b_hist)]
    dpdθg_hist = [dpdθ_hist[i][:, length(θs)+1:end] for i in eachindex(f_b_hist)]

    net_fx_hist = zeros(length(f_b_hist))
    dnext_fxdθg_hist = [zeros(1, length(θg)) for _ in eachindex(f_b_hist)]

    for i in eachindex(f_b_hist)[2:end]
        fi_b = f_b_hist[i]
        fi_x_b = fi_b[1:normalized_boundary.nodes]
        net_fx_hist[i] = sum(fi_x_b)
        dnext_fxdθg_hist[i] = dfdfk_b(fi_b)*df_bdθ_hist[i][:, length(θs)+1:end]
    end

    period = 1/(2*α)
    valid_t = t_hist .>= t_hist[end] - period

    t_avg = dt / (t_hist[valid_t][end]-t_hist[valid_t][1])

    net_fx = t_avg*sum(net_fx_hist[valid_t])
    net_fxdθg = t_avg .* sum(dnext_fxdθg_hist[valid_t])

    save_file = joinpath(DATADIR, "foil_gait_optimization_BFGS_data_test.jld2")
    jldsave(save_file; dudθg_hist, dpdθg_hist, net_fxdθg)
    
    return net_fx

end
function g_gait(θ_normalized)

    data = load(joinpath(DATADIR, "foil_gait_optimization_BFGS_data_test.jld2"))
    dudθg_hist = data["dudθg_hist"]
    dpdθg_hist = data["dpdθg_hist"]
    net_fxdθg = data["net_fxdθg"]

    return dudθg_hist, dpdθg_hist, net_fxdθg

end

function f_coshapegait(fluid::FSIModel, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θ_normalized;
    dt=0.001, t0=0.0, tf=0.01, λ1=1e-6, tol=1e-12, verbose=false)

    θs_normalized = [θ_normalized[1]]
    θg_normalized = θ_normalized[2:end]

    α = α_normalized / (fluid.ref_L / fluid.ref_u)
    θs = unnormalize_θs(DiamondFoil(;nodes=4), θs_normalized, fluid.ref_L, fluid.ref_u)
    θg = unnormalize_θg(DiamondFoil(;nodes=4), θg_normalized, fluid.ref_L, fluid.ref_u)

    boundary = Aquarium.DiamondFoil(ρ_b, θs[1], c; leading_ratio = leading_ratio, nodes=nodes)

    x = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], θg, α, t0:dt:tf)
    
    normalized_fluid = Aquarium.normalize(fluid)
    normalized_boundary = Aquarium.normalize(boundary, fluid.ref_L)

    x_normalized = map((x) -> Aquarium.normalize(boundary, x, fluid.ref_L, fluid.ref_u), x)
    
    uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
    t_hist, _, _, _, f_b_hist, dudθ_hist, dpdθ_hist, df_bdθ_hist = simulate_diff(
        normalized_fluid, normalized_boundary, uk, pk, fk_b, x_normalized, θs_normalized, θg_normalized;
        λ1=λ1, tol=tol, t0=t0, tf=tf, alg=:pardiso, iter_refine=false, verbose=verbose, α=α_normalized
    )

    net_fx_hist = zeros(length(f_b_hist))
    dnext_fxdθ_hist = [zeros(1, length(θ_normalized)) for _ in eachindex(f_b_hist)]

    for i in eachindex(f_b_hist)[2:end]
        fi_b = f_b_hist[i]
        fi_x_b = fi_b[1:normalized_boundary.nodes]
        net_fx_hist[i] = sum(fi_x_b)
        dnext_fxdθ_hist[i] = dfdfk_b(fi_b)*df_bdθ_hist[i]
    end

    period = 1/(2*α)
    valid_t = t_hist .>= t_hist[end] - period

    t_avg = dt / (t_hist[valid_t][end]-t_hist[valid_t][1])

    net_fx = t_avg*sum(net_fx_hist[valid_t])
    net_fxdθ = t_avg .* sum(dnext_fxdθ_hist[valid_t])

    save_file = joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data_test.jld2")
    jldsave(save_file; dudθ_hist, dpdθ_hist, net_fxdθ)
    
    return net_fx

end
function g_coshapegait(θ_normalized)

    data = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data_test.jld2"))
    dudθ_hist = data["dudθ_hist"]
    dpdθ_hist = data["dpdθ_hist"]
    net_fxdθ = data["net_fxdθ"]

    return dudθ_hist, dpdθ_hist, net_fxdθ

end

include("../examples/ICRA_final/foil_utils.jl")

#################################################################
## define Aquarium.x_θg_jacobian to be imported x_θg_jacobian
#################################################################

Aquarium.x_θg_jacobian(model, θg, t; kwargs...) = x_θg_jacobian(model, θg, t; kwargs...)
Aquarium.x_b_θs_jacobian(model, θs, x; kwargs...) = x_b_θs_jacobian(model, θs, x)

##############################
## define fluid variables
##############################

# time step
dt = 0.005
t0 = 0.0
tf = 0.5
T = 0:dt:tf

# fluid properties
ρ = 10.0 # kg/m^3
μ = 0.01 # Pa*s

# fluid grid
L_x = 1.0
L_y = 1.0

ne_x = 50
ne_y = 50

# boundary conditions
u_∞ = 3.0

u_west_bc = SA[u_∞, 0.0]
u_east_bc = SA[u_∞, 0.0]
u_north_bc = SA[u_∞, 0.0]
u_south_bc = SA[u_∞, 0.0]

# outflow
outflow = SA[false, true, false, false]

########################################
## define Immersed boundary variables
########################################

# boundary solid density
ρ_b = 1.0 # kg/m^3

# geometric properties
c = 0.2
leading_ratio = 0.4
a = asin(leading_ratio)-pi/180
cm_x = 0.5
cm_y = L_y/2
nodes = 30

# normalization references
ref_L = 2*c*sin(a)
ref_u = u_∞

##############################
## make models
##############################

# create boundary
boundary = Aquarium.DiamondFoil(ρ_b, a, c; leading_ratio = leading_ratio, nodes=nodes)

# make FSIModel
fluid = FSIModel(dt, ρ, μ, L_x, L_y, ref_L, ref_u, ne_x, ne_y, u_west_bc,
    u_east_bc, u_north_bc, u_south_bc, outflow; normalize=false
)
Re = round(fluid.Re)

##############################
## create oscillating motion
##############################

# frequency
α = 3.0 # Hz

# heave amplitude
A = 0.1 # meters

# pitch amplitude, phase
B = 0.0 # radians
ϕ_pitch = pi/2 # radians
ϕd_pitch = round(Int, rad2deg(ϕ_pitch))

θg = [A, B, ϕ_pitch]
θs = [a]
θ = vcat(θs, θg)
X = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], θg, α, T)

###############################################
## normalize
###############################################

α_normalized = α * (ref_L / ref_u)
θg_normalized = normalize_θg(boundary, θg, ref_L, ref_u)
θs_normalized = normalize_θs(boundary, θs, ref_L, ref_u)
θ_normalized = normalize_θ(boundary, θ, ref_L, ref_u)

normalized_fluid = Aquarium.normalize(fluid)
normalized_boundary = Aquarium.normalize(boundary, ref_L)

x_normalized = [Aquarium.normalize(boundary, X[i], ref_L, ref_u) for i in eachindex(X)]

##############################
## dynamics rollout
##############################

uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
t_hist, x_hist, u_hist, p_hist, f_b_hist = simulate(normalized_fluid, normalized_boundary, 
    uk, pk, fk_b, x_normalized; λ1=1e-6, tol=1e-12, t0=t0, tf=tf, alg=:pardiso, iter_refine=false, verbose=false
)

##############################
## save data
##############################

save_file = joinpath(DATADIR, "full_jacobians_test_wc.jld2")

jldsave(save_file; normalized_fluid, normalized_boundary, ref_L, ref_u, t_hist, x_hist, u_hist, p_hist, f_b_hist)

##############################
## load data
##############################

data = load(joinpath(DATADIR, "full_jacobians_test_wc.jld2"))
t_hist = data["t_hist"]
x_hist = data["x_hist"]
u_hist = data["u_hist"]
p_hist = data["p_hist"]
f_b_hist = data["f_b_hist"]

tk = t_hist[1]
xk = x_hist[1]
uk = u_hist[1]
pk = p_hist[1]
fk_b = f_b_hist[1]
x_bk = boundary_state(normalized_boundary, xk)

xn = x_hist[2]
un = u_hist[2]
pn = p_hist[2]
fn_b = f_b_hist[2]
tn = t_hist[2]

net_fx_hist = zeros(length(f_b_hist))

for i in eachindex(f_b_hist)[2:end]
    fi_b = f_b_hist[i]
    fi_x_b = fi_b[1:normalized_boundary.nodes]
    net_fx_hist[i] = sum(fi_x_b)
end

period = 1/(2*α)
valid_t = t_hist .>= t_hist[end] - period
t_avg = dt / (t_hist[valid_t][end]-t_hist[valid_t][1])

net_fx_true = t_avg*sum(net_fx_hist[valid_t])

#######################################################
## test individual gradients/jacobians
#######################################################

# Objective gradient wrt fk_b
∂f∂fk_b = vec(Matrix(dfdfk_b(fk_b)))
∂f∂fk_b_true = ForwardDiff.gradient(x->sum_force(x), fk_b)

@test ∂f∂fk_b == ∂f∂fk_b_true

# E jacobian
E = boundary_coupling(normalized_fluid, normalized_boundary, boundary_state(normalized_boundary, xk))
@test E*ones(size(E, 2)) ≈ ones(size(E, 1))

xk_b = boundary_state(normalized_boundary, xk)

∂vecE∂x_b = boundary_coupling_jacobian(normalized_fluid, normalized_boundary, xk_b)
∂vecE∂x_b_true = ForwardDiff.jacobian(x -> vec(boundary_coupling(normalized_fluid, normalized_boundary, x)), xk_b)
@test ∂vecE∂x_b ≈ ∂vecE∂x_b_true rtol = 1e-12

minimum(abs.(∂vecE∂x_b_true[abs.(∂vecE∂x_b_true) .> 0]))
minimum(abs.(∂vecE∂x_b[abs.(∂vecE∂x_b) .> 0]))

∂vecE_t∂x_b = boundary_coupling_transpose_jacobian(normalized_fluid, normalized_boundary, xk_b)
∂vecE_t∂x_b_true = ForwardDiff.jacobian(x -> vec(boundary_coupling(normalized_fluid, normalized_boundary, x)'), xk_b)
@test ∂vecE_t∂x_b ≈ ∂vecE_t∂x_b_true

minimum(abs.(∂vecE_t∂x_b_true[abs.(∂vecE_t∂x_b_true) .> 0]))
minimum(abs.(∂vecE_t∂x_b[abs.(∂vecE_t∂x_b) .> 0]))

# test boundary state to boundary node state jacobian
∂x_b∂x_true = sparse(ForwardDiff.jacobian(x -> boundary_state(normalized_boundary, x), xk))
∂x_b∂x = sparse(boundary_state_jacobian(normalized_boundary, xk))

@test ∂x_b∂x == ∂x_b∂x_true

#######################################################
## test co-shape-gait jacobians
#######################################################

net_fx_test = f1(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θ_normalized; dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false);
net_fx = f_coshapegait(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θ_normalized; dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false);
dudθ_hist, dpdθ_hist, net_fxdθ = g_coshapegait(θ_normalized);

net_fxdθ_true = FiniteDiff.finite_difference_jacobian(w -> f1(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, w;
    dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θ_normalized
);
# dudθ_true = FiniteDiff.finite_difference_jacobian(w -> f2(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, w;
#     dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θ_normalized
# );
# dpdθ_true = FiniteDiff.finite_difference_jacobian(w -> f3(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, w;
#     dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θ_normalized
# );

@test net_fx_true ≈ net_fx_test
@test net_fx_true ≈ net_fx

##
net_fxdθ
net_fxdθ_true
println("")
dudθ_hist[end]
dudθ_true
println("")
dpdθ_hist[end]
dpdθ_true

#######################################################
## test gait jacobians
#######################################################

net_fx = f_gait(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θs_normalized, θg_normalized; dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false);
dudθg_hist, dpdθg_hist, net_fxdθg = g_gait(θg_normalized);

net_fxdθg_true = FiniteDiff.finite_difference_jacobian(w -> f1(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, vcat(θs_normalized, w);
    dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θg_normalized
);
dudθg_true = FiniteDiff.finite_difference_jacobian(w -> f2(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, vcat(θs_normalized, w);
    dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θg_normalized
);
dpdθg_true = FiniteDiff.finite_difference_jacobian(w -> f3(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, vcat(θs_normalized, w);
    dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θg_normalized
);

@test net_fx_true ≈ net_fx

##
net_fxdθg
net_fxdθg_true
println("")
dudθg_hist[end]
dudθg_true
println("")
dpdθg_hist[end]
dpdθg_true

#######################################################
## test shape jacobians
#######################################################

net_fx = f_shape(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θs_normalized, θg_normalized; dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false);
dudθs_hist, dpdθs_hist, net_fxdθs = g_shape(θs_normalized);

net_fxdθs_true = FiniteDiff.finite_difference_gradient(w -> f1(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, vcat(w, θg_normalized);
    dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θs_normalized
);
dudθs_true = FiniteDiff.finite_difference_jacobian(w -> f2(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, vcat(w, θg_normalized);
    dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θs_normalized
);
dpdθs_true = FiniteDiff.finite_difference_jacobian(w -> f3(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, vcat(w, θg_normalized);
    dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false), θs_normalized
);

@test net_fx_true ≈ net_fx

##
net_fxdθs
net_fxdθs_true
println("")
dudθs_hist[end]
dudθs_true
println("")
dpdθs_hist[end]
dpdθs_true

########################
## Benchmark time
########################

@btime f1(fluid, cm_x, cm_y, a, c, nodes, α_normalized, θ_normalized; dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false)
@btime f(fluid, cm_x, cm_y, a, c, nodes, α_normalized, θ_normalized; dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-12, verbose=false)

#####################################
## Test full N-S KKT system
#####################################

# # run if save file not made
# gk = vcat(uk, pk, fk_b, xk);
# gn = vcat(un, pn, fn_b, xn);

# ∂R∂gn = sparse(FiniteDiff.finite_difference_jacobian(z -> full_fsi_residual(normalized_fluid, normalized_boundary, z, gk), gn))
# ∂R∂gk = sparse(FiniteDiff.finite_difference_jacobian(z -> full_fsi_residual(normalized_fluid, normalized_boundary, gn, z), gk))

# save_file = joinpath(DATADIR, "fsi_jacobians_test_full_residual.jld2")
# jldsave(save_file; ∂R∂gn, ∂R∂gk, gn, gk)

# data = load(joinpath(DATADIR, joinpath(DATADIR, "fsi_jacobians_test_full_residual.jld2")))
# ∂R∂gn = data["∂R∂gn"]
# ∂R∂gk = data["∂R∂gk"]
# gn = data["gn"]
# gk = data["gk"]

# nf_u = length(uk)
# m_D = length(pk)
# m_E = length(fk_b)
# m_x = length(xk)

# ∂R1∂un_true = ∂R∂gn[1:nf_u, 1:nf_u]
# ∂R1∂pn_true = ∂R∂gn[1:nf_u, nf_u+1:nf_u+m_D]
# ∂R1∂fn_b_true = ∂R∂gn[1:nf_u, nf_u+m_D+1:nf_u+m_D+m_E]
# ∂R1∂xn_true = ∂R∂gn[1:nf_u, nf_u+m_D+m_E+1:end]
# ∂c1∂un_true = ∂R∂gn[nf_u+1:nf_u+m_D, 1:nf_u]
# ∂c2∂un_true = ∂R∂gn[nf_u+m_D+1:nf_u+m_D+m_E, 1:nf_u]
# ∂c2∂xn_true = ∂R∂gn[nf_u+m_D+1:nf_u+m_D+m_E, nf_u+m_D+m_E+1:end]
# ∂R1∂uk_true = ∂R∂gk[1:nf_u, 1:nf_u]

# ∂R1∂un, ∂R1∂uk, ∂R1∂pn, ∂R1∂fn_b,
#     ∂R1∂xn_b, ∂R1∂xn, ∂c1∂un, ∂c2∂un, ∂c2∂xn_b, ∂c2∂xn = 
#     discrete_dynamics_jacobian(normalized_fluid, normalized_boundary, un, fn_b, xn, uk
# )

# @test ∂R1∂un ≈ ∂R1∂un_true rtol = 1e-6
# @test ∂R1∂pn ≈ ∂R1∂pn_true rtol = 1e-6
# @test ∂R1∂fn_b ≈ ∂R1∂fn_b_true rtol = 1e-6
# @test ∂R1∂xn ≈ ∂R1∂xn_true rtol = 1e-5
# @test ∂c1∂un ≈ ∂c1∂un_true rtol = 1e-6
# @test ∂c2∂un ≈ ∂c2∂un_true rtol = 1e-6
# @test ∂c2∂xn ≈ ∂c2∂xn_true rtol = 1e-6
# @test spzeros(m_D+m_E, m_D+m_E) == ∂R∂gn[nf_u+1:nf_u+m_D+m_E, nf_u+1:nf_u+m_D+m_E]

# @test ∂R1∂uk ≈ ∂R1∂uk_true rtol = 1e-6
# @test spzeros(nf_u, m_D+m_E+m_x) == ∂R∂gk[1:nf_u, nf_u+1:end]
# @test spzeros(m_D+m_E, nf_u+m_D+m_E+m_x) == ∂R∂gk[nf_u+1:end, :]