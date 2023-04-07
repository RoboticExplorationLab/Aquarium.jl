import Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Aquarium
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test
using Makie
using Makie.GeometryBasics
using CairoMakie
using BenchmarkTools
using Pardiso

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)

##############################
## define fluid variables
##############################

# time step
dt = 0.005

# fluid properties
ρ = 1.0 # kg/m^3
μ = 0.1 # Pa*s

# fluid grid
L_x = 4.0
L_y = 4.0

ne_x = 500
ne_y = 500

# boundary conditions
u_∞ = 20

u_west_bc = SA[u_∞, 0.0]
u_east_bc = SA[u_∞, 0.0]
u_north_bc = SA[u_∞, 0.0]
u_south_bc = SA[u_∞, 0.0]

# outflow
outflow = SA[false, true, false, false]

########################################
## define Immersed Boundary variables
########################################

# boundary solid density
ρ_b = 1.0 # kg/m^3

# geometric properties
D = 0.2
cm_x = 1.0
cm_y = L_y/2
θ = 0.0
nodes=100

# normalization references
ref_L = D
ref_u = u_west_bc[1]

##############################
## make models
##############################

# create boundary
boundary = Aquarium.Cylinder(ρ_b, D; nodes=nodes)

nboundary = Aquarium.normalize(boundary, ref_L)
unboundary = Aquarium.unnormalize(nboundary, ref_L)

x = Aquarium.normalize(boundary, SA[cm_x, cm_y, θ, 0., 0., 0.], ref_L, ref_u)
x_un = Aquarium.unnormalize(boundary, x, ref_L, ref_u)

# make FSIModel
freestream = FSIModel(dt, ρ, μ, L_x, L_y, ref_L, ref_u, ne_x, ne_y, u_west_bc,
    u_east_bc, u_north_bc, u_south_bc, outflow; normalize=false)

##############################
## test structure assignments
##############################

@test dt == freestream.dt
@test ρ == freestream.ρ
@test μ == freestream.μ
@test L_x == freestream.L_x
@test L_y == freestream.L_y
@test ref_L == freestream.ref_L
@test ref_u == freestream.ref_u
@test ne_x == freestream.ne_x
@test ne_y == freestream.ne_y
@test u_west_bc == freestream.u_west_bc
@test u_east_bc == freestream.u_east_bc
@test u_north_bc == freestream.u_north_bc
@test u_south_bc == freestream.u_south_bc
@test outflow == freestream.outflow
@test false == freestream.normalize

###############################################
## test normalization and un-normalization
###############################################

nfreestream = Aquarium.normalize(freestream)
unfreestream = Aquarium.unnormalize(nfreestream)

@test nfreestream.dt == freestream.dt / (ref_L/ref_u)
@test nfreestream.L_x == freestream.L_x / ref_L
@test nfreestream.L_y == freestream.L_y / ref_L
@test nfreestream.u_west_bc == freestream.u_west_bc / ref_u
@test nfreestream.u_east_bc == freestream.u_east_bc / ref_u
@test nfreestream.u_north_bc == freestream.u_north_bc / ref_u
@test nfreestream.u_south_bc == freestream.u_south_bc / ref_u
@test true == nfreestream.normalize

@test nboundary.m == boundary.m / ref_L^2
@test nboundary.J == boundary.J / ref_L^4
@test nboundary.cl == boundary.cl / ref_L
@test nboundary.r_b == boundary.r_b ./ ref_L
@test nboundary.ds == boundary.ds ./ ref_L
@test nboundary.R == boundary.R
@test nboundary.nodes == boundary.nodes
@test nboundary.normalize == true

@test x == [cm_x/ref_L, cm_y/ref_L, θ, 0*ref_u, 0*ref_u, 0*(ref_u/ref_L)]
@test x_un == [cm_x, cm_y, θ, 0., 0., 0.]

@test unfreestream.dt == freestream.dt
@test unfreestream.L_x == freestream.L_x 
@test unfreestream.L_y == freestream.L_y
@test unfreestream.u_west_bc == freestream.u_west_bc
@test unfreestream.u_east_bc == freestream.u_east_bc
@test unfreestream.u_north_bc == freestream.u_north_bc
@test unfreestream.u_south_bc == freestream.u_south_bc
@test false == freestream.normalize

@test unboundary.m ≈ boundary.m
@test unboundary.J ≈ boundary.J
@test unboundary.cl == boundary.cl
@test unboundary.r_b == boundary.r_b
@test unboundary.ds ≈ boundary.ds
@test unboundary.R == boundary.R
@test unboundary.nodes == boundary.nodes
@test unboundary.normalize == false

##############################
## test immersed boundary
##############################

# test transform
x_b = boundary_state(nboundary, x)
r_b = x_b[1:end÷2]
u_b = x_b[end÷2+1:end]

@test r_b ≈ nboundary.r_b + vcat(x[1].*ones(boundary.nodes), x[2].*ones(boundary.nodes))
@test u_b ≈ zeros(2*boundary.nodes)

##############################
## test FSI
##############################

solver = Pardiso.MKLPardisoSolver()
Pardiso.set_nprocs!(solver, Threads.nthreads())
Pardiso.set_matrixtype!(solver, Pardiso.REAL_NONSYM)
Pardiso.pardisoinit(solver)
Pardiso.fix_iparm!(solver, :N)
Pardiso.set_iparm!(solver, 5, 0)
Pardiso.set_iparm!(solver, 8, 50)
Pardiso.set_iparm!(solver, 10, 13)
Pardiso.set_iparm!(solver, 11, 0)
Pardiso.set_iparm!(solver, 13, 0)

# test coupling
E = boundary_coupling(nfreestream, nboundary, x_b)
@test E*ones(size(E, 2)) ≈ ones(size(E, 1))

# test dynamics
uk, pk, fk_b = initialize(nfreestream, nboundary)
un, pn, fn_b = discrete_dynamics(nfreestream, nboundary, E, x, uk, pk, fk_b; λ1=1e-6, alg=:pardiso, verbose=true, solver=solver)
discrete_dynamics!(nfreestream, nboundary, E, x, uk, pk, fk_b, uk, pk, fk_b; λ1=1e-6, alg=:pardiso, verbose=true, solver=solver)

@test uk ≈ un
@test pk ≈ pn
@test fk_b ≈ fn_b

##############################
## test QR vs Pardiso
##############################

uk, pk, fk_b = initialize(nfreestream, nboundary)
un_qr, pn_qr, fn_b_qr = discrete_dynamics(nfreestream, nboundary, E, x, uk, pk, fk_b; λ1=1e-6, alg=:qr, verbose=true)
un_par, pn_par, fn_b_par = discrete_dynamics(nfreestream, nboundary, E, x, uk, pk, fk_b; λ1=1e-6, alg=:pardiso, verbose=true, solver=solver)

@test un_qr ≈ un_par
@test fn_b_qr ≈ fn_b_par

## benchmark times
# @benchmark discrete_dynamics(nfreestream, E, x, uk, pk, fk_b; λ=1e-6, alg=:qr, verbose=true)
# @benchmark discrete_dynamics(nfreestream, E, x, uk, pk, fk_b; λ=1e-6, alg=:pardiso, verbose=true)
# @benchmark discrete_dynamics(nfreestream, E, x, uk, pk, fk_b; λ=1e-6, alg=:pardiso, iter_refine=true, verbose=true)

##############################
## test simulation
##############################

uk, pk, fk_b = initialize(nfreestream, nboundary)
t_hist, x_hist, u_hist, p_hist, f_b_hist = simulate(nfreestream, nboundary, uk, pk, fk_b, [x]; tf=0.01, alg=:pardiso, verbose=true)
simulate!(nfreestream, nboundary, uk, pk, fk_b, uk, pk, fk_b, [x]; tf=0.01, alg=:pardiso, verbose=true)

@test x_hist[end] ≈ x
@test u_hist[end] ≈ uk
@test p_hist[end] ≈ pk
@test f_b_hist[end] ≈ fk_b
@test length(x_hist) == length(u_hist)

# compare to true simulation
using JLD2
data = load(joinpath(DATADIR, "freestream_cylinder_Re40"*"_0to1p5_sec.jld2"))
u_hist_true = data["U_hist"]
f_b_hist_true = data["F_hist"]

@test f_b_hist[3] ≈ f_b_hist_true[3] .* vcat(nboundary.ds, nboundary.ds) ./ nfreestream.ρ
@test u_hist[3] ≈ u_hist_true[3]

##############################
## test averaging
##############################

@test average(nfreestream, uk) == Vector(nfreestream.FVM_ops.cv_avg[1] * uk + nfreestream.FVM_ops.cv_avg[2])