# Shared setup for Aquarium's example scripts.
#
# Every example begins with:
#
#     include(joinpath(@__DIR__, "..", "common.jl"))
#
# which activates the examples environment and provides the output-directory
# helpers below.
#
# Output location is the caller's concern, not the library's: Aquarium returns
# figures and data and writes nothing itself (see docs/adr/0003). These helpers
# give the examples one shared convention without putting path management back
# into the package.

import Pkg

Pkg.activate(@__DIR__)

# Aquarium is not registered, so a fresh clone cannot resolve it from a registry
# and `Pkg.instantiate()` alone would fail with
#     ERROR: expected package `Aquarium [573b866c]` to be registered
# Develop it from the repository root the first time, when no resolved
# environment exists yet. Once Aquarium is registered this block becomes
# unnecessary, though developing the local checkout is still the right thing for
# examples that live inside the repository -- they should exercise the working
# tree, not a released version.
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(path = normpath(joinpath(@__DIR__, ".."))))
end

Pkg.instantiate()

"""
Root directory for everything the examples write.

Set `AQUARIUM_OUTPUT` to redirect it; otherwise output lands in `examples/output`,
which is ignored by version control.
"""
const OUTPUT_ROOT = get(ENV, "AQUARIUM_OUTPUT", joinpath(@__DIR__, "output"))

_ensure_dir(path) = (mkpath(path); path)

"""
    visualization_dir(parts...)

Directory for figures and animations, created if absent.
"""
visualization_dir(parts...) = _ensure_dir(joinpath(OUTPUT_ROOT, "visualization", parts...))

"""
    data_dir(parts...)

Directory for saved simulation data, created if absent.
"""
data_dir(parts...) = _ensure_dir(joinpath(OUTPUT_ROOT, "data", parts...))

"""
    data_file(parts...)

Path to a file under the data directory. The containing directory is created;
the file itself is not.
"""
function data_file(parts...)
    _ensure_dir(joinpath(OUTPUT_ROOT, "data", parts[1:end-1]...))
    return joinpath(OUTPUT_ROOT, "data", parts...)
end
