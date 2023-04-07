import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using DiffFSI
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test
using Makie
using Makie.GeometryBasics
using CairoMakie
using BenchmarkTools

## define variables

# time step
dt = 0.001

# fluid properties
ρ = 1.0 # kg/m^3
μ = 0.05 # Pa*s

# fluid grid
L_x = 1.0
L_y = 1.0

ne_x = 10
ne_y = 10

##############################
## east test
##############################

# boundary conditions
u_∞ = 5.0

U_west_bc = SA[u_∞, 0.0]
U_east_bc = SA[1.0, 0.0]
U_north_bc = SA[u_∞, 0.0]
U_south_bc = SA[u_∞, 0.0]

# normalization references
ref_L = L_x
ref_U = U_west_bc[1]

# outflow
outflow = SA[false, true, false, false]

# make CFD Models
freestream = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_U, ne_x, ne_y, U_west_bc,
    U_east_bc, U_north_bc, U_south_bc, outflow; normalize=false)

# test structure assignments
@test dt == freestream.dt
@test ρ == freestream.ρ
@test μ == freestream.μ
@test L_x == freestream.L_x
@test L_y == freestream.L_y
@test ref_L == freestream.ref_L
@test ref_U == freestream.ref_U
@test ne_x == freestream.ne_x
@test ne_y == freestream.ne_y
@test U_west_bc == freestream.U_west_bc
@test U_east_bc == freestream.U_east_bc
@test U_north_bc == freestream.U_north_bc
@test U_south_bc == freestream.U_south_bc
@test outflow == freestream.outflow
@test false == freestream.normalize

# test normalization and un-normalization
nfreestream = DiffFSI.normalize(freestream)
unfreestream = DiffFSI.unnormalize(nfreestream)

@test nfreestream.dt == freestream.dt / (ref_L/ref_U)
@test nfreestream.L_x == freestream.L_x / ref_L
@test nfreestream.L_y == freestream.L_y / ref_L
@test nfreestream.U_west_bc == freestream.U_west_bc / ref_U
@test nfreestream.U_east_bc == freestream.U_east_bc / ref_U
@test nfreestream.U_north_bc == freestream.U_north_bc / ref_U
@test nfreestream.U_south_bc == freestream.U_south_bc / ref_U
@test true == nfreestream.normalize

@test unfreestream.dt == freestream.dt
@test unfreestream.L_x == freestream.L_x 
@test unfreestream.L_y == freestream.L_y
@test unfreestream.U_west_bc == freestream.U_west_bc
@test unfreestream.U_east_bc == freestream.U_east_bc
@test unfreestream.U_north_bc == freestream.U_north_bc
@test unfreestream.U_south_bc == freestream.U_south_bc
@test false == freestream.normalize

# test dynamics
Uk, pk = initialize(nfreestream)
Un, pn = discrete_dynamics(nfreestream, Uk, pk)
discrete_dynamics!(nfreestream, Uk, pk, Uk, pk)

@test Uk == Un
@test pk == pn

# test simulation
Uk, pk = initialize(nfreestream)
U_hist, p_hist = simulate(nfreestream, Uk, pk; tf=0.5)
simulate!(nfreestream, Uk, pk, Uk, pk; tf=0.5)

@test U_hist[end] == Uk
@test p_hist[end] == pk

# test averaging
@test average(nfreestream, Uk) == Vector(nfreestream.FVM_ops.cv_avg[1] * Uk + nfreestream.FVM_ops.cv_avg[2])

# turn into grid
uk_grid, vk_grid = fluidgrid(nfreestream, average(nfreestream, Uk))

@test minimum(uk_grid .== u_∞)
@test minimum(vk_grid .== 0.0)

## try plotting
# plot_streamlines(nfreestream, average(nfreestream, Uk))
# plot_streamlines(nfreestream, Uk)

##############################
## west test
##############################

# boundary conditions
u_∞ = -5.0

U_west_bc = SA[1.0, 0.0]
U_east_bc = SA[u_∞, 0.0]
U_north_bc = SA[u_∞, 0.0]
U_south_bc = SA[u_∞, 0.0]

# normalization references
ref_L = L_x
ref_U = U_east_bc[1]

# outflow
outflow = SA[true, false, false, false]

# make CFD Models
freestream = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_U, ne_x, ne_y, U_west_bc,
    U_east_bc, U_north_bc, U_south_bc, outflow; normalize=false)
nfreestream = DiffFSI.normalize(freestream)

# test dynamics
Uk, pk = initialize(nfreestream)
Un, pn = discrete_dynamics(nfreestream, Uk, pk)
discrete_dynamics!(nfreestream, Uk, pk, Uk, pk)

@test Uk == Un
@test pk == pn

# test simulation
Uk, pk = initialize(nfreestream)
U_hist, p_hist = simulate(nfreestream, Uk, pk; tf=0.5)
simulate!(nfreestream, Uk, pk, Uk, pk; tf=0.5)

@test U_hist[end] == Uk
@test p_hist[end] == pk

# test averaging
@test average(nfreestream, Uk) == Vector(nfreestream.FVM_ops.cv_avg[1] * Uk + nfreestream.FVM_ops.cv_avg[2])

# turn into grid
uk_grid, vk_grid = fluidgrid(nfreestream, average(nfreestream, Uk))

@test minimum(uk_grid .== u_∞)
@test minimum(vk_grid .== 0.0)

## try plotting
# plot_streamlines(nfreestream, average(nfreestream, Uk))
# plot_streamlines(nfreestream, Uk)

##############################
## north test
##############################

# boundary conditions
u_∞ = 5.0

U_west_bc = SA[0.0, u_∞]
U_east_bc = SA[0.0, u_∞]
U_north_bc = SA[0.0, 1.0]
U_south_bc = SA[0.0, u_∞]

# normalization references
ref_L = L_x
ref_U = U_south_bc[2]

# outflow
outflow = SA[false, false, true, false]

# make CFD Models
freestream = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_U, ne_x, ne_y, U_west_bc,
    U_east_bc, U_north_bc, U_south_bc, outflow; normalize=false)
nfreestream = DiffFSI.normalize(freestream)

# test dynamics
Uk, pk = initialize(nfreestream)
Un, pn = discrete_dynamics(nfreestream, Uk, pk)
discrete_dynamics!(nfreestream, Uk, pk, Uk, pk)

@test Uk == Un
@test pk == pn

# test simulation
Uk, pk = initialize(nfreestream)
U_hist, p_hist = simulate(nfreestream, Uk, pk; tf=0.5)
simulate!(nfreestream, Uk, pk, Uk, pk; tf=0.5)

@test U_hist[end] == Uk
@test p_hist[end] == pk

# test averaging
@test average(nfreestream, Uk) == Vector(nfreestream.FVM_ops.cv_avg[1] * Uk + nfreestream.FVM_ops.cv_avg[2])

# turn into grid
uk_grid, vk_grid = fluidgrid(nfreestream, average(nfreestream, Uk))

@test minimum(uk_grid .== 0.0)
@test minimum(vk_grid .== u_∞)

## try plotting
# plot_streamlines(nfreestream, average(nfreestream, Uk))
# plot_streamlines(nfreestream, Uk)

##############################
## south test
##############################

# boundary conditions
u_∞ = -5.0

U_west_bc = SA[0.0, u_∞]
U_east_bc = SA[0.0, u_∞]
U_north_bc = SA[0.0, u_∞]
U_south_bc = SA[0.0, 1.0]

# normalization references
ref_L = L_x
ref_U = U_north_bc[2]

# outflow
outflow = SA[false, false, false, true]

# make CFD Models
freestream = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_U, ne_x, ne_y, U_west_bc,
    U_east_bc, U_north_bc, U_south_bc, outflow; normalize=false)
nfreestream = DiffFSI.normalize(freestream)

# test dynamics
Uk, pk = initialize(nfreestream)
Un, pn = discrete_dynamics(nfreestream, Uk, pk)
discrete_dynamics!(nfreestream, Uk, pk, Uk, pk)

@test Uk == Un
@test pk == pn

# test simulation
Uk, pk = initialize(nfreestream)
U_hist, p_hist = simulate(nfreestream, Uk, pk; tf=0.5)
simulate!(nfreestream, Uk, pk, Uk, pk; tf=0.5)

@test U_hist[end] == Uk
@test p_hist[end] == pk

# test averaging
@test average(nfreestream, Uk) == Vector(nfreestream.FVM_ops.cv_avg[1] * Uk + nfreestream.FVM_ops.cv_avg[2])

# turn into grid
uk_grid, vk_grid = fluidgrid(nfreestream, average(nfreestream, Uk))

@test minimum(uk_grid .== 0.0)
@test minimum(vk_grid .== u_∞)

## try plotting
# plot_streamlines(nfreestream, average(nfreestream, Uk))
# plot_streamlines(nfreestream, Uk)

##############################
## diagonal test
##############################

# boundary conditions
u_∞ = 5.0

U_west_bc = SA[u_∞, u_∞]
U_east_bc = SA[1.0, 1.0]
U_north_bc = SA[1.0, 1.0]
U_south_bc = SA[u_∞, u_∞]

# normalization references
ref_L = L_x
ref_U = U_north_bc[2]

# outflow
outflow = SA[false, true, true, false]

# make CFD Models
freestream = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_U, ne_x, ne_y, U_west_bc,
    U_east_bc, U_north_bc, U_south_bc, outflow; normalize=false)
nfreestream = DiffFSI.normalize(freestream)

# test dynamics
Uk, pk = initialize(nfreestream)
Un, pn = discrete_dynamics(nfreestream, Uk, pk)
discrete_dynamics!(nfreestream, Uk, pk, Uk, pk)

@test Uk == Un
@test pk == pn

# test simulation
Uk, pk = initialize(nfreestream)
U_hist, p_hist = simulate(nfreestream, Uk, pk; tf=0.5)
simulate!(nfreestream, Uk, pk, Uk, pk; tf=0.5)

@test U_hist[end] == Uk
@test p_hist[end] == pk

# test averaging
@test average(nfreestream, Uk) == Vector(nfreestream.FVM_ops.cv_avg[1] * Uk + nfreestream.FVM_ops.cv_avg[2])

# turn into grid
uk_grid, vk_grid = fluidgrid(nfreestream, average(nfreestream, Uk))

@test minimum(uk_grid .== u_∞)
@test minimum(vk_grid .== u_∞)

## try plotting
# plot_streamlines(nfreestream, average(nfreestream, Uk))
# plot_streamlines(nfreestream, Uk)