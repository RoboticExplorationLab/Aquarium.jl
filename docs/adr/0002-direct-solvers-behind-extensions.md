# Direct sparse solvers live behind package extensions

`solver_type` accepts `:gmres`, `:backslash`, `:pardiso`, and `:mumps`. Pardiso was already a
weakdep behind `ext/PardisoExt.jl`; MUMPS and MPI were hard dependencies. We moved MUMPS and
MPI behind `ext/MumpsExt.jl` following the same pattern, so both direct solvers are now
optional.

The default is `solver_type = :gmres` with `preconditioner_type = :ilu`, and every example and
test in the repository uses `:gmres` — no call site anywhere exercises `:mumps` or `:pardiso`.
Keeping MUMPS and MPI as hard dependencies meant every user pulled an MPI binary stack, and
every CI job built it, to support a path none of them took. Extensions preserve the capability
for large off-repo cluster runs at no cost to the default install.

## Consequences

- `MPI.Init()` and `MPI.Finalize()` move out of `src/linear_solve/linear_solver.jl` and the two
  cleanup paths in `AquariumTank.jl` / `Fluid.jl` into the extension.
- A `MUMPS_LOADED` `Ref` and function stubs guard the `:mumps` branch, mirroring
  `PARDISO_LOADED`. Requesting `:mumps` without `using MUMPS` raises an actionable error rather
  than a `MethodError`.
- Anyone relying on `:mumps` must add `using MUMPS` to their script. This is a breaking change
  for such users and is why it lands before the first registered release rather than after.
