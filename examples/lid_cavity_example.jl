import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Aquarium
using LinearAlgebra
using SparseArrays
using StaticArrays
using Makie
using Makie.GeometryBasics
using CairoMakie
using JLD2
using Interpolations
using Tensors

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)

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
u_west_bc = SA[0.0, 0.0]
u_east_bc = SA[0.0, 0.0]
u_north_bc = SA[5.0, 0.0]
u_south_bc = SA[0.0, 0.0]

# normalization references
ref_L = L_x
ref_u = u_north_bc[1]

# outflow
outflow = SA[false, false, false, false]

##############################
## define CFD model
##############################

fluid = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_u, ne_x, ne_y, u_west_bc,
    u_east_bc, u_north_bc, u_south_bc, outflow; normalize=false)

##############################
## normalize
##############################

normalized_fluid = Aquarium.normalize(fluid)

##############################
## simulate
##############################

uk, pk = initialize(normalized_fluid)
t_hist, u_hist, p_hist = simulate(normalized_fluid, uk, pk; λ=1e-6, tol=1e-6, tf=tf,
    iter_refine=false, verbose=false)

##############################
## save data
##############################

mkdir(joinpath(DATADIR, "lid_cavity"))
save_file = joinpath(DATADIR, "lid_cavity", "lid_cavity_0to1_sec.jld2")
jldsave(save_file; normalized_fluid, ref_L, ref_u, t_hist, u_hist, p_hist)

##############################
## load data
##############################

data = load(joinpath(DATADIR, "lid_cavity", "lid_cavity_0to1_sec.jld2"))
t_hist = data["t_hist"]
u_hist = data["u_hist"]
p_hist = data["p_hist"]
uk = u_hist[end]

##############################
## plot
##############################

uk_avg = average(normalized_fluid, uk)

scene = plot_streamlines(normalized_fluid, uk_avg; density=100)
display(scene)

scene = plot_vorticity(normalized_fluid, uk_avg; levels=100)
display(scene)