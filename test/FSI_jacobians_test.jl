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
using ForwardDiff
using FiniteDiff
using Pardiso
using ProgressMeter
using Rotations
using JLD2

const DATADIR = expanduser(joinpath("~/Aquarium", "data"))
const VISDIR = expanduser(joinpath("~/Aquarium", "visualization"))

mkpath(DATADIR)
mkpath(VISDIR)

############################
## helper functions
############################

function vec_E(model, boundary, x_b)
    vec_Et = vec(boundary_coupling(model, boundary, x_b))
    return vec_Et
end
function vec_E_transpose(model, boundary, x_b)
    vec_Et = vec(boundary_coupling(model, boundary, x_b)')
    return vec_Et
end

function boundary_coupling_x(model, boundary, x)
    x_b = BoundaryModels.boundary_state(boundary, x)
    E = boundary_coupling(model, boundary, x_b)

    return E
end

function full_fsi_residual(model::FSIModel, boundary::ImmersedBoundary,
    gk::AbstractVector, gkm1::AbstractVector, fext=[0, 0])

    nf_u = (model.ne_x - 1) * model.ne_y + model.ne_x * (model.ne_y - 1)
    m_D = model.ne_x * model.ne_y
    m_E = 2 * boundary.nodes
    m_x = length(gk) - nf_u - m_D - m_E

    uk = gk[1:nf_u]
    pk = gk[nf_u+1:nf_u+m_D]
    fk_b_tilda = gk[nf_u+m_D+1:nf_u+m_D+m_E]
    xk = gk[end-m_x+1:end]

    ukm1 = gkm1[1:nf_u]
    pkm1 = gkm1[nf_u+1:nf_u+m_D]
    fkm1_b_tilda = gkm1[nf_u+m_D+1:nf_u+m_D+m_E]
    xkm1 = gkm1[end-m_x+1:end]

    fk_b = fk_b_tilda .* (model.ρ * model.h_x * model.h_y)
    fkm1_b = fkm1_b_tilda .* (model.ρ * model.h_x * model.h_y)

    r1_eval = Aquarium.R1(model, boundary, uk, pk, fk_b, xk, ukm1)
    c1_eval = Aquarium.c1(model, uk)
    c2_eval = Aquarium.c2(model, boundary, uk, xk)
    r2_eval = Aquarium.R2(boundary, xk, fk_b, xkm1, fext; dt=model.dt)

    return vcat(r1_eval, c1_eval, c2_eval, r2_eval)
    
end

##############################
## define fluid variables
##############################

# time step
t = 0.0
tf = 0.02
dt = 0.005

# fluid properties
ρ = 1.0 # kg/m^3
μ = 0.1 # Pa*s

# fluid grid
L_x = 4.0
L_y = 4.0

ne_x = 50
ne_y = 50

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
l = 0.2
w_effective = 0.01
cm_x = 1.0
cm_y = L_y/2
θ = pi/2
nodes=5

x = SA[cm_x, cm_y, θ, 0., 0., 0.]

# normalization references
ref_L = l
ref_u = u_west_bc[1]

##############################
## make models
##############################

# create boundary
boundary = Aquarium.Bar1D(ρ_b, l; w_effective=w_effective, nodes=nodes)

# make FSIModel
fluid = FSIModel(dt, ρ, μ, L_x, L_y, ref_L, ref_u, ne_x, ne_y, u_west_bc,
    u_east_bc, u_north_bc, u_south_bc, outflow; normalize=false
)

##############################
## normalize
##############################

normalized_boundary = Aquarium.normalize(boundary, ref_L)
x_normalized = Aquarium.normalize(boundary, x, ref_L, ref_u)
normalized_fluid = Aquarium.normalize(fluid)

##############################
## dynamics rollout
##############################

uk, pk, fk_b = initialize(normalized_fluid, normalized_boundary)
t_hist, x_hist, u_hist, p_hist, f_b_hist = simulate(normalized_fluid, normalized_boundary, 
    uk, pk, fk_b, [x_normalized]; λ1=1e-6, tol=1e-12, t0=t, tf=0.01, alg=:pardiso, iter_refine=false, verbose=true
)

tk = t_hist[3]
xk = x_hist[3]
uk = u_hist[3]
pk = p_hist[3]
fk_b = f_b_hist[3]

xkm1 = x_hist[2]
ukm1 = u_hist[2]
pkm1 = p_hist[2]
fkm1_b = f_b_hist[2]

##############################
## test E
##############################

E = boundary_coupling(normalized_fluid, normalized_boundary, boundary_state(normalized_boundary, xk))
@test E*ones(size(E, 2)) ≈ ones(size(E, 1))

# vec(E) transpose jacobian
xk_b = boundary_state(normalized_boundary, xk)

∂vecE∂x_b = boundary_coupling_jacobian(normalized_fluid, normalized_boundary, xk_b)
∂vecE∂x_b_true = ForwardDiff.jacobian(x -> vec(boundary_coupling(normalized_fluid, normalized_boundary, x)), xk_b)
@test ∂vecE∂x_b ≈ ∂vecE∂x_b_true rtol = 1e-16

minimum(abs.(∂vecE∂x_b_true[abs.(∂vecE∂x_b_true) .> 0]))
minimum(abs.(∂vecE∂x_b[abs.(∂vecE∂x_b) .> 0]))

# vec(E^T) jacobian
∂vecE_t∂x_b = boundary_coupling_transpose_jacobian(normalized_fluid, normalized_boundary, xk_b)
∂vecE_t∂x_b_true = ForwardDiff.jacobian(x -> vec(boundary_coupling(normalized_fluid, normalized_boundary, x)'), xk_b)
@test ∂vecE_t∂x_b ≈ ∂vecE_t∂x_b_true

minimum(abs.(∂vecE_t∂x_b_true[abs.(∂vecE_t∂x_b_true) .> 0]))
minimum(abs.(∂vecE_t∂x_b[abs.(∂vecE_t∂x_b) .> 0]))

#######################################################
## test boundary state to boundary node state jacobian
#######################################################

∂x_b∂x_true = sparse(ForwardDiff.jacobian(x -> boundary_state(normalized_boundary, x), xk))
∂x_b∂x = sparse(boundary_state_jacobian(normalized_boundary, xk))

@test ∂x_b∂x == ∂x_b∂x_true

#######################################################
## test FSI gradients/jacobian
#######################################################

# define variables

ds = vcat(normalized_boundary.ds, normalized_boundary.ds)
xk_b = boundary_state(normalized_boundary, xk)
uk_b = xk_b[end÷2+1:end]

G = normalized_fluid.FVM_ops.G
L = normalized_fluid.FVM_ops.L[1]
L_bc = normalized_fluid.FVM_ops.L[3]
D_bc = normalized_fluid.FVM_ops.D[2]

dt = normalized_fluid.dt
Re = normalized_fluid.Re

nf_u = length(uk)
m_D = length(pk)
m_E = length(fk_b)
m_x = length(xk)

# define kkt system submatrices
A = (1/dt).*(sparse(I, nf_u, nf_u) - (dt/(2*Re)).*L)
r(z) = (1/dt).*(sparse(I, nf_u, nf_u) + (dt/(2*Re)).*L)*z - 0.5.*N(normalized_fluid, z)
bc1 = (1/Re).*L_bc
bc2 = -D_bc
bc3 = sparse(-uk_b)

# calculate jacobians
∂R1∂uk, ∂R1∂ukm1, ∂R1∂pk, ∂R1∂fk_b,
    ∂R1∂xk, ∂c1∂uk, ∂c2∂uk, ∂c2∂xk =
    discrete_dynamics_jacobian(normalized_fluid, normalized_boundary, uk, fk_b, xk, ukm1
)

## 

@test N_jacobian(normalized_fluid, uk) ≈ ForwardDiff.jacobian(x -> N(normalized_fluid, x), uk)
@test ∂R1∂ukm1 ≈ -FiniteDiff.finite_difference_jacobian(z -> r(z), ukm1) rtol = 1e-6
@test ∂R1∂pk ≈ G
@test ∂R1∂fk_b ≈ FiniteDiff.finite_difference_jacobian(z -> R1(normalized_fluid, normalized_boundary, uk, pk, z, xk, ukm1), fk_b) rtol = 1e-6

∂R1∂xk_true = ForwardDiff.jacobian(z -> R1(normalized_fluid, normalized_boundary, uk, pk, fk_b, z, ukm1), xk)
@test ∂R1∂xk ≈ ∂R1∂xk_true

minimum(abs.(∂R1∂xk_true[abs.(∂R1∂xk_true) .> 0]))
minimum(abs.(∂R1∂xk[abs.(∂R1∂xk) .> 0]))

@test ∂c1∂uk ≈ G'
@test ∂c2∂uk ≈ E

∂c2∂xk_true = ForwardDiff.jacobian(z -> c2(normalized_fluid, normalized_boundary, uk, z), xk)
@test ∂c2∂xk ≈ ∂c2∂xk_true

minimum(abs.(∂c2∂xk_true[abs.(∂c2∂xk_true) .> 0]))
minimum(abs.(∂c2∂xk[abs.(∂c2∂xk) .> 0]))

# fk_b_tilda = fk_b ./ (normalized_fluid.ρ * normalized_fluid.h_x * normalized_fluid.h_y)
# ∂R1∂xk2 = (kron(sparse(fk_b_tilda'), sparse(I, nf_u, nf_u)) * (∂vecE_t∂x_b_true * ∂x_b∂x))
# ∂R1∂xk_true = FiniteDiff.finite_difference_jacobian(z -> R1(normalized_fluid, normalized_boundary, uk, pk, fk_b, z, ukm1), xk)

# @test ∂R1∂xk2 ≈ ∂R1∂xk_true rtol = 1e-5

#######################################################
## Check residual
#######################################################

fk_b_tilda = fk_b ./ (normalized_fluid.ρ * normalized_fluid.h_x * normalized_fluid.h_y)
fkm1_b_tilda = fkm1_b ./ (normalized_fluid.ρ * normalized_fluid.h_x * normalized_fluid.h_y)

gk = vcat(uk, pk, fk_b_tilda, xk)
gkm1 = vcat(ukm1, pkm1, fkm1_b_tilda, xkm1)

residual = full_fsi_residual(normalized_fluid, normalized_boundary, gk, gkm1)

r1_eval = residual[1:nf_u]
c1_eval = residual[nf_u+1:nf_u+m_D]
c2_eval = residual[nf_u+m_D+1:nf_u+m_D+m_E]
r2_eval = residual[end-m_x+1:end]

@test maximum(abs.(r1_eval)) < 1e-12 
@test maximum(abs.(c1_eval)) < 1e-12
@test maximum(abs.(c2_eval)) < 1e-12

#######################################################
## Check full kkt matrix
#######################################################

# kkt_true = sparse(FiniteDiff.finite_difference_jacobian(z -> full_fsi_residual(normalized_fluid, normalized_boundary, z, gkm1, [0, 0]), gk))

# save_file = joinpath(DATADIR, "fsi_jacobians_test_full_kkt.jld2")
# jldsave(save_file; kkt_true, gk, gkm1)

data = load(joinpath(DATADIR, joinpath(DATADIR, "fsi_jacobians_test_full_kkt.jld2")))
kkt_true = data["kkt_true"]
gk = data["gk"]
gkm1 = data["gkm1"]

∂R1∂uk, ∂R1∂ukm1, ∂R1∂pk, ∂R1∂fk_b,
    ∂R1∂xk, ∂c1∂uk, ∂c2∂uk, ∂c2∂xk = 
    discrete_dynamics_jacobian(normalized_fluid, normalized_boundary, uk, fk_b_tilda .* (normalized_fluid.ρ * normalized_fluid.h_x * normalized_fluid.h_y), xk, ukm1
)
∂R2∂xk, ∂R2∂fk_b, ∂R2∂xkm1 =
    discrete_dynamics_jacobian(normalized_boundary, xk, fk_b_tilda .* (normalized_fluid.ρ * normalized_fluid.h_x * normalized_fluid.h_y), [0, 0];
        dt=normalized_fluid.dt
)

kkt = vcat(hcat(∂R1∂uk, ∂c1∂uk', ∂c2∂uk', ∂R1∂xk),
    hcat(∂c1∂uk, spzeros(m_D, m_D), spzeros(m_D, m_E), spzeros(m_D, m_x)),
    hcat(∂c2∂uk, spzeros(m_E, m_D), spzeros(m_E, m_E), ∂c2∂xk),
    hcat(spzeros(m_x, nf_u), spzeros(m_x, m_D), ∂R2∂fk_b.*(normalized_fluid.ρ * normalized_fluid.h_x * normalized_fluid.h_y), ∂R2∂xk)
)

@test ∂R1∂uk ≈ kkt_true[1:nf_u, 1:nf_u] rtol = 1e-6
@test vcat(∂c1∂uk, ∂c2∂uk) ≈ kkt_true[nf_u+1:nf_u+m_D+m_E, 1:nf_u] rtol = 1e-6
@test hcat(∂c1∂uk', ∂c2∂uk') ≈ kkt_true[1:nf_u, nf_u+1:nf_u+m_D+m_E] rtol = 1e-6
@test ∂R1∂xk ≈ kkt_true[1:nf_u, nf_u+m_D+m_E+1:end] rtol = 1e-5
@test spzeros(m_D, m_x) == kkt_true[nf_u+1:nf_u+m_D, nf_u+m_D+m_E+1:end]
@test ∂c2∂xk ≈ kkt_true[nf_u+m_D+1:nf_u+m_D+m_E, nf_u+m_D+m_E+1:end] rtol = 1e-6
@test ∂R2∂fk_b.*(normalized_fluid.ρ * normalized_fluid.h_x * normalized_fluid.h_y) ≈ kkt_true[end-m_x+1:end, nf_u+m_D+1:nf_u+m_D+m_E] rtol = 1e-5
@test ∂R2∂xk ≈ kkt_true[end-m_x+1:end, end-m_x+1:end] rtol = 1e-5