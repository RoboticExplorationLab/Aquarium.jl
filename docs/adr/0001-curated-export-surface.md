# Curated export surface instead of `@exportAll`

Aquarium previously exported every name defined in the module via `ExportAll.@exportAll()`,
producing 270 exported names — of which 123 were referenced nowhere outside their defining
file and one was `eval`. We replaced this with an explicit `export` list seeded from the 58
names that `examples/` and `test/` actually use. Everything else remains reachable as
`Aquarium.foo`.

## Considered Options

Keeping `@exportAll` was seriously considered. It is genuinely convenient: new functions are
available immediately with no export list to maintain and no drift between the list and
reality. Since we release as `0.x`, the usual semver argument against it is weak — breaking
changes only need a minor bump.

## Consequences

The decisive reasons were documentation and collisions, not semver:

- **Documenter.** `checkdocs = :exports` errors on every exported name lacking a docstring.
  With 270 exports and ~9 docstrings, publishing docs would have meant writing 261 docstrings
  or setting `checkdocs = :none`, which silently omits most of the API from the docs.
- **Namespace collisions.** `RigidBody` and `Joint` are the core exported types of
  RigidBodyDynamics.jl — the package an Aquarium user is most likely to load alongside it.
  `Bar`, `Pendulum`, `Fluid`, `rotation_matrix`, `body_velocity`, and `linear_solve!` are
  similarly generic.

Costs we accepted: the 89 internals referenced only by inline `@testitem` blocks must now be
called as `Aquarium.foo` inside those blocks, touching most of the 97 blocks. The `ExportAll`
dependency is dropped.
