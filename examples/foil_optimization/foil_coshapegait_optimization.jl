import Pkg
Pkg.activate(joinpath(@__DIR__,"..", ".."))

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
using LBFGSB
using Rotations
using BenchmarkTools

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)
include("foil_utils.jl")

##############################
## Useful Functions
##############################

function f(fluid::FSIModel, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, θ_normalized;
    dt=0.001, t0=0.0, tf=0.01, λ1=1e-6, tol=1e-12, verbose=false)

    data_file = joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data"*replace(join(θ_normalized), "." => "")*".jld2")

    if isfile(data_file)

        data = load(data_file)
        net_fx = data["net_fx"]

    else

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
        t_hist, _, _, _, f_b_hist, _, _, df_bdθ_hist = simulate_diff(
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

        jldsave(data_file; boundary, fluid, θ_normalized, net_fx, net_fxdθ, t_hist, f_b_hist, df_bdθ_hist)

    end

    return net_fx

end
function g(θ_normalized)

    data = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data"*replace(join(θ_normalized), "." => "")*".jld2"))
    net_fxdθ = vec(data["net_fxdθ"])

    return net_fxdθ

end

function dfdfk_b(fk_b)

    u1 = ones(length(fk_b)÷2)
    u2 = zeros(length(fk_b)÷2)

    u = vcat(u1, u2)

    ∂f∂fk_b = sparse(u'*sparse(I, length(fk_b), length(fk_b)))

    return ∂f∂fk_b

end

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
L_x = 3.0
L_y = 3.0

ne_x = 300
ne_y = 300

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
cm_x = 1.0
cm_y = L_y/2
nodes = 56

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
Bd = round(Int, rad2deg(B))
ϕd_pitch = round(Int, rad2deg(ϕ_pitch))

X = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], [A, B, ϕ_pitch], α, T)

θ_initial = [a, A, B, ϕ_pitch]

# specify bounds
θ_l = [pi/30, 0.05, 0.0, 0.0]
θ_b = [asin(leading_ratio)-pi/180, 0.15, pi/3, pi]

# mkpath(joinpath(VISDIR, "diamondfoil"))
# anime_file = joinpath(VISDIR, "diamondfoil", "diamondfoil_oscillating_$α"*"Hz_$A"*"m_$Bd"*"deg_$ϕd_pitch"*"phase.mp4")
# Aquarium.animate_boundary(boundary, anime_file, T, X; x_lim=[0.5, 2.75], y_lim=[0.75, 2.25],
#     color=:black, lengthscale=1.5, framerate=30, timescale=10, show_vel=false
# )

###############################################
## normalize
###############################################

θ_initial_normalized = normalize_θ(boundary, θ_initial, ref_L, ref_u)
θ_l_normalized = normalize_θ(boundary, θ_l, ref_L, ref_u)
θ_b_normalized = normalize_θ(boundary, θ_b, ref_L, ref_u)

α_normalized = α * (ref_L / ref_u)
normalized_fluid = Aquarium.normalize(fluid)
normalized_boundary = Aquarium.normalize(boundary, ref_L)

##########################################################
## define single input objective and gradient functions
##########################################################

obj(w) = f(fluid, ρ_b, cm_x, cm_y, c, leading_ratio, nodes, α_normalized, w; dt=dt, t0=t0, tf=tf, λ1=1e-6, tol=1e-10, verbose=false);
g!(z, w) = z .= g(w)

##########################################################
## test objective and gradient functions
##########################################################

z = zeros(length(θ_initial))

net_thrust_initial = obj(θ_initial_normalized)
g!(z, θ_initial_normalized)

@show(z)

###############################################
## run BFGS
###############################################

optimizer = L_BFGS_B(4, 10)

n = 4  # the dimension of the problem
x = θ_initial_normalized  # the initial guess

# set up bounds
bounds = zeros(3, 4)

for i in 1:length(θ_initial)
    bounds[1,i] = 2
    bounds[2,i] = copy(θ_l_normalized[i])
    bounds[3,i] = copy(θ_b_normalized[i])
end

# net_thrust_optimal, θ_optimal_normalized = optimizer(obj, g!, x, bounds, m=5, factr=1e7, pgtol=1e-5, iprint=101, maxfun=200, maxiter=200);

# @show(net_thrust_optimal)
# @show(θ_optimal_normalized)

@time optimizer(obj, g!, x, bounds, m=5, factr=1e7, pgtol=1e-5, iprint=101, maxfun=200, maxiter=200);

###############################################
## Save Results
###############################################

θ_optimal = unnormalize_θ(boundary, θ_optimal_normalized, ref_L, ref_u)

save_file = joinpath(DATADIR, "diamondfoil_example_coshapegait_optimization_optimal_results.jld2")
jldsave(save_file; dt, t0, tf, α, ρ_b, c, leading_ratio, cm_x, cm_y, nodes, fluid, normalized_fluid, θ_initial, θ_initial_normalized,
    net_thrust_initial, θ_l, θ_b, θ_l_normalized, θ_b_normalized, θ_optimal, θ_optimal_normalized, net_thrust_optimal
)

###############################################
## Save BFGS history
###############################################

data = load(joinpath(DATADIR, "diamondfoil_example_coshapegait_optimization_optimal_results.jld2"))
dt = data["dt"]
t0 = data["t0"]
tf = data["tf"]
α = data["α"]
ρ_b = data["ρ_b"]
c = data["c"]
leading_ratio = data["leading_ratio"]
cm_x = data["cm_x"]
cm_y = data["cm_y"]
nodes = data["nodes"]
fluid = data["fluid"]
normalized_fluid = data["normalized_fluid"]
θ_initial = data["θ_initial"]
θ_initial_normalized = data["θ_initial_normalized"]
θ_l = data["θ_l"]
θ_b = data["θ_b"]
θ_l_normalized = data["θ_l_normalized"]
θ_b_normalized = data["θ_b_normalized"]
θ_optimal = data["θ_optimal"]
θ_optimal_normalized = data["θ_optimal_normalized"]
net_thrust_optimal = data["net_thrust_optimal"]
net_thrust_initial = data["net_thrust_initial"]

data_iter1 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data02562100734711923504960249310267874604989214475798240415707963267948966.jld2"))

θ_iter1_normalized = data_iter1["θ_normalized"]
θ_iter1 = unnormalize_θ(data_iter1["boundary"], θ_iter1_normalized, fluid.ref_L, fluid.ref_u)
net_thrust_iter1 = data_iter1["net_fx"]

data_iter5 = load(joinpath(DATADIR, "foil_coshapegait_optimization_BFGS_data010471975511965977097670574165226220386518836301494615045257988223608.jld2"))

θ_iter5_normalized = data_iter5["θ_normalized"]
θ_iter5 = unnormalize_θ(data_iter5["boundary"], θ_iter5_normalized, fluid.ref_L, fluid.ref_u)
net_thrust_iter5 = data_iter5["net_fx"]

save_file = joinpath(DATADIR, "diamondfoil_example_coshapegait_optimization_history.jld2")
jldsave(save_file; dt, t0, tf, α, ρ_b, c, leading_ratio, cm_x, cm_y, nodes, fluid, normalized_fluid, θ_initial, θ_initial_normalized,
    net_thrust_initial, θ_l, θ_b, θ_l_normalized, θ_b_normalized, θ_optimal, θ_optimal_normalized, net_thrust_optimal,
    θ_iter1_normalized, θ_iter1, net_thrust_iter1, θ_iter5_normalized, θ_iter5, net_thrust_iter5
)