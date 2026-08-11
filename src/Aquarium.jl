module Aquarium

# Flag to check if Pardiso extension is loaded
const PARDISO_LOADED = Ref(false)

# Pardiso wrapper function stubs (implemented in PardisoExt when Pardiso is loaded)
function create_pardiso_solver end
function pardiso_set_nprocs! end
function pardiso_set_matrixtype! end
function pardiso_init! end
function pardiso_fix_iparm! end
function pardiso_set_iparm! end
function pardiso_set_phase! end
function pardiso_factorize! end
function pardiso_solve! end

abstract type SolidBody end
abstract type Actuator end
abstract type Controller end
abstract type DeformableBody <: SolidBody end
abstract type AbstractRigidBody <: SolidBody end
abstract type SolidSystem end

## Imports
using Base.Threads
using ExportAll
using LinearAlgebra
using SparseArrays
using SymRCM
using AMD
using Metis
using Krylov
using MPI
using MUMPS
using IncompleteLU
using ILUZero
using AlgebraicMultigrid
using Rotations
using ForwardDiff
using ProgressMeter
using CairoMakie
using TestItems

# Parallel utilities (composable task-based threading helpers)
include("parallel_utils.jl")

# Linear solve
include("linear_solve/preconditioners.jl")
include("linear_solve/linear_solver.jl") 

# Fluid components (FVMGrid must come before Fluid)
include("fluid/FVMGrid.jl")
include("fluid/fvm_operators.jl")
include("fluid/fvm_grid_utils.jl")
include("fluid/Fluid.jl")
include("fluid/differentiable_params.jl")

# Solid components — topical subdirectories, matching BeyondReachDOJO layout.
# Ordering is load-time dependency ordering: shapes → bodies → joints → systems
# (struct+constructor) → actuators → dynamics → differentiable_params → concrete
# system constructors → simulate → legacy adapters.

# Shapes
include("solid/shapes/Shape.jl")
include("solid/shapes/Bar.jl")
include("solid/shapes/Disc.jl")

# Bodies (rotation_utils is a body-frame math helper)
include("solid/bodies/rotation_utils.jl")
include("solid/bodies/RigidBody.jl")

# Joints
include("solid/joints/Joint.jl")
include("solid/joints/PinJoint.jl")
include("solid/joints/WorldPinJoint.jl")

# System topology (must come before PassiveSystem/ActuatedSystem)
include("solid/SystemTopology.jl")

# System struct definitions (plain structs + constructor + validation)
include("solid/systems/PassiveSystem.jl")
include("solid/systems/ActuatedSystem.jl")
include("solid/systems/NoSystem.jl")

# Actuators (require PDController → JointServoMotor dispatches on ActuatedSystem)
include("solid/actuators/PDController.jl")
include("solid/actuators/JointServoMotor.jl")

# Shared solid-system dynamics (state layout, PE, residuals, jacobians).
# Must come after the struct definitions but before differentiable_params.
include("solid/dynamics.jl")

# Differentiable parameter collect/inject
include("solid/differentiable_params.jl")

# Boundary-state kinematics — input to the FSI immersed-boundary kernels
include("solid/boundary_state.jl")

# Plotting
include("solid/plot_solid_system.jl")

# Concrete system constructor functions (aquarium zoo)
include("aquarium_zoo/Pendulum.jl")
include("aquarium_zoo/DoublePendulum.jl")
include("aquarium_zoo/Eel.jl")
include("aquarium_zoo/ActuatedPendulum.jl")
include("aquarium_zoo/RExEel.jl")
include("aquarium_zoo/FreeBar.jl")
include("aquarium_zoo/FreeDisc.jl")

# Standalone solid-system time-stepping driver
include("solid/simulate.jl")

# Fluid-solid interaction (require Fluid and AbstractRigidBody types)
include("fluid_solid_interaction/discrete_deltas.jl")
include("fluid_solid_interaction/original_immersed_boundary_method.jl")
include("fluid_solid_interaction/weak_form_immersed_boundary_method.jl")
include("fluid_solid_interaction/no_slip_constraint.jl")

# Main simulation environment (requires all above)
include("AquariumTank.jl")

# Visualization (requires all simulation components)
include("visualization/aquarium_plots.jl")
include("visualization/aquarium_animations.jl")

## Export all functions
@exportAll()

# Aquarium owns no filesystem paths and creates no directories on load. It
# returns figures and data; where they go is the caller's decision. See
# docs/adr/0003-library-owns-no-filesystem-paths.md -- in particular, two of the
# directory constants that used to live here targeted the package's own source
# tree, which is read-only for any registry install.

end