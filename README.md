# Aquarium.jl

[![Build Status](https://github.com/RoboticExplorationLab/Aquarium.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/RoboticExplorationLab/Aquarium.jl/actions/workflows/CI.yml?query=branch%3Amain)

A differentiable fluid-structure interaction solver for robotics applications.

Aquarium couples a 2D finite-volume fluid to multi-rigid-body systems through immersed-boundary
no-slip constraints, and solves the coupled system monolithically so that gradients flow through
the coupling. That makes it usable not just for simulation but for trajectory optimization and
design optimization of bodies moving in fluid.

For the results of the studies for our RSS 2026 paper, please refer to this repo: https://github.com/RoboticExplorationLab/RoboticSwimmingWithUnifiedFluidRobot

## Aquarium V0.2.0

This is the new open-source, refactored version of Aquarium, based on our accepted RSS paper. It
features:

- Strongly-coupled multi-rigid-body–fluid interaction based on unified discrete variational
  mechanics
- Differentiability improvements that allow interfacing with more optimization solvers, not just
  L-BFGS-B
- Bug fixes in the 2D FVM, particularly boundary conditions

It will be presented at JuliaCon 2026 as part of the Computational Physics Minisymposium.

## Installation

Requires Julia 1.10 or later.

```julia
using Pkg
Pkg.add("Aquarium")
```

Until registration in the General registry completes, install directly from this repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/RoboticExplorationLab/Aquarium.jl")
```

[Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl) is an optional dependency. Load it before
Aquarium to enable the `:pardiso` linear solver and preconditioner; everything works without it,
using the default GMRES solver with an incomplete-LU preconditioner.

## Quick start

A damped, spring-loaded pendulum, integrated with Aquarium's variational integrator:

```julia
using Aquarium

time_step = 0.01
pendulum = Pendulum(time_step;
    bar_length = 0.5,
    mass       = 5.0,
    moi        = (1 / 12) * 5.0 * 0.5^2,   # thin rod about its centre of mass
    stiffness  = 5.0,                      # N·m/rad
    damping    = 1.0,                      # N·m·s/rad
)

# Aquarium works in maximal coordinates; build the initial state from a joint angle.
initial_configuration = pendulum_maximal_from_minimal(pendulum, [deg2rad(-45)])
initial_state = initialize_solid_state(pendulum, vcat(initial_configuration, zeros(3)))

trajectories = simulate_solid_system(pendulum, initial_state, 5.0)

time_traj = trajectories[:time_traj]
configuration_traj = trajectories[:configuration_traj]
```

For the fluid-structure coupling Aquarium exists for, build an `AquariumTank` from a `Fluid` and
one or two solid systems and call `simulate_aquarium`. See `examples/` — those runs take minutes
rather than seconds, so they are kept out of the quick start.

## Examples

The `examples/` directory contains solid-only examples, fluid-only examples, coupled
fluid-structure simulations, and the case studies from the papers.

```
julia examples/solid_examples/pendulum_example.jl
```

Examples compute and plot but write nothing by default. To save artifacts, set any of
`AQUARIUM_SAVE_DATA`, `AQUARIUM_SAVE_FIGURES`, `AQUARIUM_SAVE_ANIMATIONS`, or `AQUARIUM_SAVE_ALL`
to `true`. Output goes to `examples/output/` unless `AQUARIUM_OUTPUT` redirects it.

## Citing

If you use Aquarium.jl and the Aquarium algorithm as part of your research, teaching, or other
activities, we would be grateful if you could cite our works:

[1] J. H. Lee, M. Y. Michelis, R. Katzschmann and Z. Manchester, "Aquarium: A Fully Differentiable
Fluid-Structure Interaction Solver for Robotics Applications," 2023 IEEE International Conference
on Robotics and Automation (ICRA), London, United Kingdom, 2023, pp. 11272-11279, doi:
10.1109/ICRA48891.2023.10161494.

```
@INPROCEEDINGS{10161494,
  author={Lee, Jeong Hun and Michelis, Mike Y. and Katzschmann, Robert and Manchester, Zachary},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Aquarium: A Fully Differentiable Fluid-Structure Interaction Solver for Robotics Applications}, 
  year={2023},
  volume={},
  number={},
  pages={11272-11279},
  doi={10.1109/ICRA48891.2023.10161494}}
```

[2] J. H. Lee, J. Hu, S. Kwok, C. Majidi, and Z. Manchester, "Realizing Robotic Swimming with
Unified Fluid-Robot Multiphysics," in 2026 Robotics: Science and Systems, 2026. [Online].
Available: https://arxiv.org/abs/2506.05012

```
@misc{lee2026realizingroboticswimmingunified,
      title={Realizing Robotic Swimming with Unified Fluid-Robot Multiphysics}, 
      author={Jeong Hun Lee and Junzhe Hu and Sofia Kwok and Carmel Majidi and Zachary Manchester},
      year={2026},
      eprint={2506.05012},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.05012}, 
}
```

## License

MIT. See [LICENSE](LICENSE).
