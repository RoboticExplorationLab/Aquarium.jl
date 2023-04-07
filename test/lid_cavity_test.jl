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

##############################
## define fluid variables
##############################

# time step
dt = 0.001

# fluid properties
ρ = 1.0 # kg/m^3
μ = 0.05 # Pa*s

# fluid grid
L_x = 1.0
L_y = 1.0

ne_x = 3
ne_y = 3

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
## make CFD model
##############################

lc = CFDModel(dt, ρ, μ, L_x, L_y, ref_L, ref_U, ne_x, ne_y, U_west_bc,
    U_east_bc, U_north_bc, U_south_bc, outflow; normalize=false)

##############################
## test structure assignments
##############################

@test dt == lc.dt
@test ρ == lc.ρ
@test μ == lc.μ
@test L_x == lc.L_x
@test L_y == lc.L_y
@test ref_L == lc.ref_L
@test ref_U == lc.ref_U
@test ne_x == lc.ne_x
@test ne_y == lc.ne_y
@test U_west_bc == lc.U_west_bc
@test U_east_bc == lc.U_east_bc
@test U_north_bc == lc.U_north_bc
@test U_south_bc == lc.U_south_bc
@test outflow == lc.outflow
@test false == lc.normalize

###############################################
## test normalization and un-normalization
###############################################

nlc = DiffFSI.normalize(lc)
unlc = DiffFSI.unnormalize(nlc)

@test nlc.dt == lc.dt / (ref_L/ref_U)
@test nlc.L_x == lc.L_x / ref_L
@test nlc.L_y == lc.L_y / ref_L
@test nlc.U_west_bc == lc.U_west_bc / ref_U
@test nlc.U_east_bc == lc.U_east_bc / ref_U
@test nlc.U_north_bc == lc.U_north_bc / ref_U
@test nlc.U_south_bc == lc.U_south_bc / ref_U
@test true == nlc.normalize

@test unlc.dt == lc.dt
@test unlc.L_x == lc.L_x 
@test unlc.L_y == lc.L_y
@test unlc.U_west_bc == lc.U_west_bc
@test unlc.U_east_bc == lc.U_east_bc
@test unlc.U_north_bc == lc.U_north_bc
@test unlc.U_south_bc == lc.U_south_bc
@test false == lc.normalize

##############################
## test dynamics
##############################

Uk, pk = initialize(nlc)
Un, pn = discrete_dynamics(nlc, Uk, pk; λ=1e-6, tol=1e-12)
discrete_dynamics!(nlc, Uk, pk, Uk, pk; λ=1e-6, tol=1e-12)

@test Uk == Un
@test pk == pn

##############################
## test qr vs pardiso
##############################

Uk, pk = initialize(nlc)
Un_qr, pn_qr = discrete_dynamics(nlc, Uk, pk; λ=1e-6, tol=1e-12, alg=:qr);
Un_par, pn_par = discrete_dynamics(nlc, Uk, pk; λ=1e-6, tol=1e-12, alg=:lu);

@test Un_qr ≈ Un_par
@test pn_qr ≈ pn_par rtol=1e-2

##############################
## test simulation
##############################

Uk, pk = initialize(nlc)
T_hist, U_hist, p_hist = simulate(nlc, Uk, pk; tf=0.5)
simulate!(nlc, Uk, pk, Uk, pk; tf=0.5)

@test U_hist[end] == Uk
@test p_hist[end] == pk

##############################
## test averaging
##############################

@test average(nlc, Uk) == Vector(nlc.FVM_ops.cv_avg[1] * Uk + nlc.FVM_ops.cv_avg[2])