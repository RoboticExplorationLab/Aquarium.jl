module Aquarium

function x_θg_jacobian end
function x_b_θs_jacobian end

export FSIModel, normalize, unnormalize, normalize_θs, unnormalize_θs, initialize, simulate, simulate_diff,
    simulate!, discrete_dynamics, discrete_dynamics_diff, discrete_dynamics!,
    discrete_dynamics_diff!, N, boundary_coupling, discrete_delta, N_jacobian,
    discrete_dynamics_jacobian, boundary_coupling_jacobian,
    boundary_coupling_transpose_jacobian, boundary_coupling_with_jacobian,
    discrete_delta_jacobian, R1, c1, c2, x_θg_jacobian, x_b_θs_jacobian

export CFDModel, normalize, unnormalize, initialize, simulate,
    simulate!, discrete_dynamics, discrete_dynamics!

export ImmersedBoundary
export RigidBody
    
export FVM_CDS_2D, boundary_ind
export animate_vorticity, plot_vorticity, vorticity,
    animate_streamlines, plot_streamlines,
    animate_velocityfield, plot_velocityfield,
    U_interpolate, fluidgrid, fluidcoord, meshgrid,
    stack_states, average

using LinearAlgebra
using SparseArrays
using StaticArrays
using Interpolations
using Rotations
using Pardiso
using ProgressMeter
using FiniteDiff
using ForwardDiff
using BlockDiagonals
using Base.Threads
using CairoMakie
using GeometryBasics
using Colors

include("FVM_CDS_2D.jl")
include("CFD.jl")
include("ImmersedBoundary.jl")
include("RigidBody.jl")
include("FSI.jl")
include("FSI_utils.jl")
include("boundary_models/boundary_models.jl")

export Cylinder, Multilink1D, SRLFishTail1D, SRLFishTail2D,
    Bar1D, Shuttlecock, DiamondFoil
export sort_points, normalize, unnormalize,
    boundary_state, boundary_state!, boundary_state_jacobian,
    boundary_state_jacobian!, body_force, body_force_jacobian,
    body_force_jacobian!, dynamics, dynamics!, dynamics_jacobian,
    dynamics_jacobian!, discrete_dynamics_jacobian,
    discrete_dynamics_jacobian!, simulate_predefined, plot_boundary,
    plot_boundary!
export maximal_state, maximal_state!, maximal_state_inertial, maximal_state_inertial!,
    maximal_body_to_inertial, maximal_inertial_to_body
end