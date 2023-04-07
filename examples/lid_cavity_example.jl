import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Aquarium
using LinearAlgebra
using SparseArrays
using StaticArrays
using Makie
using Makie.GeometryBasics
using CairoMakie
using MATLAB
using JLD2
using Interpolations
using Tensors

##############################
## define fluid environment
##############################

# time step
dt = 0.01
tf = 1.0

# fluid properties
ρ = 1.0 # kg/m^3
μ = 0.05 # Pa*s

# fluid grid
L_x = 1.0
L_y = 1.0

ne_x = 50
ne_y = 50

# boundary conditions
U_west_bc = SA[0.0, 0.0]
U_east_bc = SA[0.0, 0.0]
U_north_bc = SA[5.0, 0.0]
U_south_bc = SA[0.0, 0.0]

# normalization references
ref_L = L_x
ref_U = U_north_bc[1]

# outflow
outflow = SA[false, false, false, false]

##############################
## define CFD model
##############################

fluid = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_U, ne_x, ne_y, U_west_bc,
    U_east_bc, U_north_bc, U_south_bc, outflow; normalize=false)

##############################
## normalize
##############################

normalized_fluid = DiffFSI.normalize(fluid)

##############################
## simulate
##############################

Uk, pk = initialize(normalized_fluid)
T_hist, U_hist, p_hist = simulate(normalized_fluid, Uk, pk; λ=1e-6, tol=1e-6, tf=tf,
    iter_refine=false, verbose=false)

##############################
## save data
##############################

save_file = joinpath(DiffFSI.DATADIR, "lid_cavity_0to1_sec.jld2")
jldsave(save_file; normalized_fluid, ref_L, ref_U, T_hist, U_hist, p_hist)

##############################
## load data
##############################

data = load(joinpath(DiffFSI.DATADIR, "lid_cavity_0to1_sec.jld2"))
T_hist = data["T_hist"]
U_hist = data["U_hist"]
p_hist = data["p_hist"]
Uk = U_hist[end]

##############################
## plot
##############################

Uk_avg = average(normalized_fluid, Uk)

plot_streamlines(normalized_fluid, Uk_avg; density=100)
plot_vorticity(normalized_fluid, Uk_avg; levels=100)

##############################
## create animation
##############################

anime_file = joinpath(DiffFSI.FIGDIR, "lid_cavity",
    "lid_cavity_Re100_streamlines.mp4")

animate_streamlines(normalized_fluid, T_hist, U_hist, anime_file;
    density=100.0, framerate=20, timescale=5.0, display_live=false)

anime_file = joinpath(DiffFSI.FIGDIR, "lid_cavity",
    "lid_cavity_Re100_vorticity.mp4")

animate_vorticity(normalized_fluid, T_hist[1:20], U_hist[1:20], anime_file;
    levels=100, framerate=20, timescale=5.0, display_live=false)