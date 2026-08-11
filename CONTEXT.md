# Aquarium

A differentiable fluid-structure interaction solver: a 2D finite-volume fluid coupled to
multi-rigid-body systems through immersed-boundary no-slip constraints, solved as one
monolithic system so that gradients flow through the coupling.

## Language

### The coupled system

**Aquarium Tank**:
The complete coupled simulation: one fluid, one bluff body, and one swimmer, together with the
index layout that maps them into a single state vector.
_Avoid_: environment, scene, world, simulation

**Fluid**:
The 2D finite-volume velocity and pressure field on a staggered grid.
_Avoid_: flow, domain, CFD state

**Swimmer**:
The solid system whose motion is *solved for* — its body state lives in the aquarium state
vector and is coupled to the fluid through no-slip duals.
_Avoid_: robot, body, agent, actuated system

**Bluff Body**:
The solid system whose motion is *prescribed externally* — it contributes a no-slip constraint
to the fluid but no state to the aquarium state vector.
_Avoid_: obstacle, static body, boundary

Swimmer and bluff body are **roles, not types**. Both slots hold a `SolidSystem`; an unused
slot holds `NoSystem`. What distinguishes them is whether their motion is solved or imposed.

### Solid mechanics

**Solid System**:
A rigid-body assembly — bodies, joints, and optionally actuators — that occupies the swimmer or
bluff-body role. Concretely `PassiveSystem`, `ActuatedSystem`, or `NoSystem`.
_Avoid_: mechanism, multibody, linkage, robot

**Minimal Coordinates**:
The joint-space description of a solid system's configuration — one number per joint degree of
freedom.
_Avoid_: generalized coordinates, joint angles, reduced coordinates

**Maximal Coordinates**:
The per-body description of a solid system's configuration — position and orientation for every
body, with joints enforced as explicit constraints.
_Avoid_: full coordinates, world coordinates, Cartesian coordinates

**Aquarium Zoo**:
The library of ready-made solid systems shipped with Aquarium — `Pendulum`, `DoublePendulum`,
`Eel`, `RExEel`, `FreeBar`, `FreeDisc`, and their actuated variants.
_Avoid_: models, examples, presets, gallery

### Coupling

**No-Slip Constraint**:
The condition equating fluid velocity to solid boundary velocity at the immersed boundary. It
is what couples the two physics, and it is enforced as a constraint rather than a penalty.
_Avoid_: boundary condition, interface condition, coupling term

**Dual**:
A Lagrange multiplier in the monolithic KKT system. The state vector carries fluid duals,
swimmer duals, and a no-slip dual for each of the swimmer and bluff body.
_Avoid_: multiplier, adjoint, costate

**Immersed Boundary Method**:
The scheme spreading solid boundary forces onto the fluid grid and interpolating fluid velocity
back. Aquarium implements two: the **original** method, which acts at discrete boundary nodes,
and the **weak-form** method, which integrates over boundary segments.
_Avoid_: IBM (in prose), fictitious domain, embedded boundary

**Boundary State**:
The positions and velocities of a solid system's boundary nodes — the solid-side input to the
immersed-boundary kernels.
_Avoid_: surface state, interface state, marker points

### Differentiability

**Differentiable Params**:
The subset of a tank's physical parameters exposed to optimization, moved in and out of the
tank as a flat vector by `collect_differentiable_params` and `inject_differentiable_params`.
_Avoid_: design variables, tunable parameters, hyperparameters
