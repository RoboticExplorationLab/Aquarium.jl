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
using BenchmarkTools

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)

include("foil_optimization/foil_utils.jl")

##############################
## define fluid variables
##############################

# time step
dt = 0.005
t0 = 0.0
tf = 0.6
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
leading_ratio = 0.3
θ = pi/20
cm_x = 1.0
cm_y = L_y/2
nodes = 52

# normalization references
ref_L = 2*c*sin(θ)
ref_u = u_∞

##############################
## make models
##############################

# create boundary
boundary = Aquarium.DiamondFoil(ρ_b, θ, c; leading_ratio = leading_ratio, nodes=nodes)

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
A = 0.15 # meters

# pitch amplitude, phase
B = pi/3 # radians
ϕ_pitch = pi/2 # radians
Bd = round(Int, rad2deg(B))
ϕd_pitch = round(Int, rad2deg(ϕ_pitch))

X = oscillating_motion_x(boundary, [cm_x, cm_y, 0.0], [A, B, ϕ_pitch], α, T)

##############################
## Animate boundary motion
##############################

mkpath(joinpath(VISDIR, "diamondfoil"))

anime_file = joinpath(VISDIR, "diamondfoil", "diamondfoil_oscillating_$α"*"Hz_$A"*"m_$Bd"*"deg_$ϕd_pitch"*"phase.mp4")
Aquarium.animate_boundary(boundary, anime_file, T, X; x_lim=[0.5, 2.75], y_lim=[0.75, 2.25],
    color=:black, lengthscale=1.5, framerate=30, show_vel=false
)

###############################################
## normalize
###############################################

α_normalized = α * (ref_L / ref_u)
θ_normalized = normalize_θg(boundary, [A, B, ϕ_pitch], ref_L, ref_u)
normalized_fluid = Aquarium.normalize(fluid)
normalized_boundary = Aquarium.normalize(boundary, ref_L)

X = [Aquarium.normalize(boundary, X[i], ref_L, ref_u) for i in eachindex(X)]

##############################
## simulate
##############################

uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
t_hist, x_hist, u_hist, p_hist, f_b_hist = simulate(normalized_fluid, normalized_boundary, uk, pk, fk_b, X;
     λ1=1e-6, tol=1e-12, tf=tf, alg=:pardiso, verbose=false
)

##############################
## save data
##############################

save_file = joinpath(DATADIR, "oscillating_diamondfoil_$Re"*"Re_$α"*"Hz_$A"*"m_$Bd"*"deg_$ϕd_pitch"*"phase.jld2")
jldsave(save_file; normalized_fluid, normalized_boundary, ref_L, ref_u, t_hist, x_hist, u_hist, p_hist, f_b_hist, df_bdθ_hist)

##############################
## load data
##############################

data = load(joinpath(DATADIR, "oscillating_diamondfoil_$Re"*"Re_$α"*"Hz_$A"*"m_$Bd"*"deg_$ϕd_pitch"*"phase.jld2"))
t_hist = data["t_hist"]
x_hist = data["x_hist"]
u_hist = data["u_hist"]
p_hist = data["p_hist"]
f_b_hist = data["f_b_hist"]
df_bdθ_hist = data["df_bdθ_hist"]
x = x_hist[end]
uk = u_hist[end]

##############################
## calculate forces
##############################

Fx = ones(length(f_b_hist))
Fy = ones(length(f_b_hist))

for i in eachindex(f_b_hist)
    fi_b = f_b_hist[i]
    normalized_Fx = fi_b[1:normalized_boundary.nodes]    
    Fx[i] = -sum(normalized_Fx) .* fluid.ref_u^2 .* fluid.ref_L^2
end

##############################
## plot forces
##############################
fig, ax, sp = lines(t_hist[5:end], Fx[5:end],
    color=:blue, axis = (xlabel = "Time (s)", ylabel = "Thrust (N/m)")
);
display(fig)

##############################
## plot
##############################

uk_avg = average(normalized_fluid, uk)
scene = plot_streamlines(normalized_fluid, normalized_boundary, x, uk_avg;
    density=150, x_lim=[0.5, 2.75], y_lim=[0.75, 2.25],
    colormap=:blues, fontsize=25)
display(scene)

scene = plot_vorticity(normalized_fluid, normalized_boundary, x, uk_avg;
    levels=100, x_lim=[0.5, 2.75], y_lim=[0.75, 2.25], level_perc=0.2,
    colormap=:bwr, fontsize=25)
display(scene)

##############################
## create animation
##############################

mkpath(joinpath(VISDIR, "diamondfoil"))

anime_file = joinpath(VISDIR, "diamondfoil",
    "oscillating_diamondfoil_$Re"*"Re_$α"*"Hz_$A"*"m_$Bd"*"deg_$ϕd_pitch"*"phase_streamlines.mp4")

animate_streamlines(normalized_fluid, normalized_boundary, t_hist[1:end-1], x_hist[1:end-1], u_hist, anime_file;
    density=150.0, x_lim=[0.5, 2.75], y_lim=[0.75, 2.25],
    framerate=20, timescale=10.0, display_live=false)

anime_file = joinpath(VISDIR, "diamondfoil",
    "oscillating_diamondfoil_$Re"*"Re_$α"*"Hz_$A"*"m_$Bd"*"deg_$ϕd_pitch"*"phase_vorticity.mp4")

animate_vorticity(normalized_fluid, normalized_boundary, t_hist[1:end-1], x_hist[1:end-1], u_hist, anime_file;
    levels=20, x_lim=[0.5, 2.75], y_lim=[0.75, 2.25], level_perc=0.2, colormap=:bwr, fontsize=25,
    framerate=20, timescale=10.0, display_live=false)