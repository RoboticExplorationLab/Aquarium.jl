# Tasks

## Todo

### Canonical differentiable interface via `ChainRulesCore.rrule` on `simulate_aquarium`
- **Labels:** feature, differentiability, autodiff, api

Make Aquarium a canonical differentiable function that any ChainRules-aware
AD consumer (Zygote, Enzyme, ReverseDiff, SciMLSensitivity, Optimization.jl
with `AutoZygote`, DifferentiationInterface.jl, ...) can plug into without
needing any knowledge of Aquarium's internal solver, stage-objective closure
interface, or per-parameter-block gradient fields.

**Why this matters.** Aquarium's current gradient interface — pass
`calculate_stage_objective` / `calculate_stage_objective_gradients` closures
into `simulate_aquarium`, read gradients back from a result struct — is an
Aquarium-specific convention. The canonical Julia interface is
`ChainRulesCore.rrule`; the equivalents in other ecosystems are
`jax.custom_vjp` (JAX) and `torch.autograd.Function` (PyTorch). All of these
let a library declare "I am a differentiable black box, here is my pullback"
and let AD frameworks plug in by convention. Until Aquarium provides an
`rrule`, downstream SciML packages cannot consume it as a differentiable
building block — a user writing
`OptimizationFunction((p,_) -> loss(simulate_aquarium(inject(tank,p),...)), AutoZygote())`
will fail because Zygote has no pullback to find.

Note that `AutoForwardDiff()` is *not* an alternative path: it requires
propagating `ForwardDiff.Dual` numbers through the Newton/Krylov solver,
which we have explicitly decided against. The canonical ChainRules hook
(`rrule`) is the only viable plug-and-play entry point, regardless of
parameter count.

**Approach (forward-mode sensitivity assembly, wrapped in rrule).** Keep the
Float64 Newton/Krylov solver as-is. Define
`ChainRulesCore.rrule(::typeof(simulate_aquarium), tank, state_0, T; kwargs...)`.

- **Forward pass**: run the primal solver AND propagate `dx_k/dp` alongside
  it stage-by-stage. At each stage, solve `KKT · (dx_{k+1}/dp) = RHS_k`,
  which reuses the already-factored primal KKT matrix *in the forward
  direction* (no transpose). Assemble the full Jacobian
  `J = ∂trajectory/∂p` during the forward pass and store it in the pullback
  closure.

- **Pullback**: one matvec. `p_bar = J^T · traj_bar`. No adjoint-through-time
  walk; no stage-by-stage cotangent propagation.

- **Sidesteps the KKT time-stepping blocker**: the variational integrator's
  KKT matrix has a row/column mismatch that makes the *transposed* (adjoint)
  direction ill-defined. Forward sensitivities only use the primal direction,
  which is exactly what the current solver already does — so this approach
  avoids the blocker entirely.

- **Cost**: `O(|p|)` extra work per stage in the forward pass. Cheap at 10s–
  100s of params (current use cases); intractable at NN scale (thousands of
  weights). For the NN-scale case, see the reverse-mode adjoint task below.

**Interface changes required.**

- `simulate_aquarium` must return a plain trajectory — no more
  result-struct-with-embedded-gradients, no `fluid_properties=p` kwarg
  escape hatch.
- The closure interface (`calculate_stage_objective` /
  `calculate_stage_objective_gradients`) is removed from the public API.
  Users write their loss in their own code:
  `loss(simulate_aquarium(tank, state_0, T))`.
- Existing internal callers (ILC, optimization scripts) migrate to the new
  interface.

**Unblocked by:** The FSI/Fluid refactor (current PR sequence) — specifically
the tank-level `collect/inject_differentiable_params` API, the removal of
the `fluid_properties=p` kwarg escape hatch, and the typed `AquariumTank{S}`.

**Not blocked by the KKT time-stepping mismatch** — the forward-mode
sensitivity assembly approach sidesteps it. Only the reverse-mode adjoint
task below is blocked by the mismatch.

**Success criterion.** A passing end-to-end integration test where an
Optimization.jl problem with `AutoZygote()` optimizes a small (10–50 param)
physics closure through `simulate_aquarium`, with no Aquarium-specific
gradient plumbing in the user code:

```julia
using Optimization, OptimizationOptimJL, Zygote

function loss(p, _)
    tank_p = inject_differentiable_params(tank, p)
    trajectory = simulate_aquarium(tank_p, state_0, T)
    swimming_loss(trajectory)
end

opt_func = OptimizationFunction(loss, AutoZygote())
prob = OptimizationProblem(opt_func, p0)
sol = solve(prob, BFGS())
```

If this runs and the loss decreases, the `rrule` is considered done for its
primary use case. Follow-up coverage (Enzyme, DifferentiationInterface,
higher-order derivatives) is incremental and tracked separately.

### Reverse-mode adjoint-through-time for NN-scale param counts
- **Labels:** performance, differentiability, autodiff

Replace the forward-mode sensitivity assembly in the `rrule` above with a
reverse-mode adjoint-through-time walk. Required for scaling to NN-sized
parameter counts (thousands of weights) — forward-mode sensitivity cost
scales with `|p|`, reverse-mode adjoint cost is `O(1)` in `|p|`.

**Status.** Not urgent. The forward-mode sensitivity approach (rrule task
above) covers all current SciML use cases (10s–100s of params). Becomes
mandatory once a neural controller feeds `swimmer_control` from a Flux
network, or any other large-param-block scenario.

**Approach.** Replace the forward pass's sensitivity propagation with
caching of stage Jacobians only, and replace the pullback's single matvec
with an adjoint sweep: for `k = T, T-1, ..., 1`, propagate
`x_bar_k ← (∂x_{k+1}/∂x_k)^T · x_bar_{k+1}` and accumulate `p_bar` via the
existing stage VJP machinery (`calculate_no_slip_constraint_jacobian`,
`calculate_fluid_stationarity_jacobian`, etc.).

**Blocked by:** The variational-integrator KKT time-stepping mismatch. The
adjoint walk needs the *transposed* KKT operator, which is ill-defined due
to a row/column mismatch in the current integrator formulation. This needs
to be characterized and either worked around or fixed at the integrator
level.

**Suggested first step.** Write a minimal reproducer showing the time-step
mismatch on a 1-body, 1-fluid example. Then decide whether to fix the
integrator or pay a correction cost inside the pullback.

## Done

### Hand-coded analytical Jacobian for `∂_∂x_kp1` of solid stationarity
- **Labels:** accuracy, differentiability

Replaced the `ForwardDiff.jacobian` call for the `∂_∂kp1` block in both
`calculate_solid_stationarity_jacobian` and `calculate_solid_dynamics_jacobian`
with hand-coded analytical assembly (PE Hessian, damping Jacobian, actuator
Jacobian with clamp handling, forward constraint Jacobians, prescribed-angle
constraint Jacobians). Also replaced three ForwardDiff calls in
`calculate_no_slip_constraint_vjp_jacobian` (midpoint Jacobian, original and
weak-form VJP Hessians) with analytical versions.

Solid `∂_∂kp1` now matches ForwardDiff at `atol=1e-12`. Aquarium-level
`∂D/∂x_kp1` tightened from `rtol=1e-3` to `atol=1e-6` (remaining ~1e-7
noise from evaluation-order differences in the no-slip VJP coupling path).
~11% faster on both solid dynamics Jacobian and no-slip VJP Jacobian.

### Kinematic constraint mode for actuated joints (prescribed motion)
- **Labels:** feature, solid, actuators, differentiability

`actuation_mode::Symbol` (`:pd` / `:prescribed`) on `ActuatedSystem`. In
`:prescribed` mode, joint angles are imposed as holonomic constraints via
Lagrange multipliers; the multiplier is the required joint torque. The
existing `PinJoint`/`WorldPinJoint` constraint machinery is reused — see
`calculate_prescribed_angle_constraint_residual` and the `_add_prescribed_angle_constraint_vjp!`
family in [src/solid/dynamics.jl](src/solid/dynamics.jl).

Success-criterion test: [test/prescribed_vs_pd_trajopt_tests.jl](test/prescribed_vs_pd_trajopt_tests.jl)
runs a swing-up trajectory optimization on an `ActuatedPendulum` with
`Kp = 1e3` in both modes and confirms that `:prescribed` reaches a lower
final cost in the same iteration budget under identical backtracking
gradient descent. Recovered Lagrange-multiplier torques are bounded and
physically reasonable in steady state.

### Three-point discrete delta kernel as a selectable option in both IB methods
- **Labels:** feature, fsi, immersed-boundary

### Migrate test suite to TestItemRunner with inline `@testitem` blocks
- **Labels:** testing, infrastructure, refactor
