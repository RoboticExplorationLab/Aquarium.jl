# The library owns no filesystem paths

Loading Aquarium used to create directories: `src/Aquarium.jl` defined `EXAMPLES_DIR`,
`TEST_DIR`, `VIS_DIR`, and `DATA_DIR` and called `mkpath` on all four at module scope, creating
`~/aquarium/visualization` and `~/aquarium/data` in the user's home directory as a side effect
of `using Aquarium`. We removed all four constants and all four `mkpath` calls. Aquarium
returns figures and data; the caller decides where they go.

Two concrete failures motivated this beyond principle. `mkpath(EXAMPLES_DIR)` and
`mkpath(TEST_DIR)` target the package's own source tree, which is read-only when a package is
installed from a registry — the normal case in CI, on clusters, and for every user who is not
developing Aquarium. And the `~/aquarium` location was hardcoded, so users had no way to
redirect output.

## Consequences

The 13 example scripts that referenced `Aquarium.VIS_DIR` / `Aquarium.DATA_DIR` now get their
output directory from `examples/common.jl`, which reads `ENV["AQUARIUM_OUTPUT"]` and defaults
to a gitignored `examples/output/`. This keeps a shared convention for the examples without
putting path management in the library's API.

A future contributor may be tempted to re-add a convenience output directory to the package.
That is what this record exists to prevent: the boundary is deliberate — a solver library has
no business writing to `$HOME`.
