# Shared setup for Aquarium's example scripts.
#
# Every example begins with:
#
#     include(joinpath(@__DIR__, "..", "common.jl"))
#
# which activates the examples environment and provides the output helpers and
# artifact flags below.
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

# `save` and `jldsave` are deliberately NOT imported here.
#
# This file is included into Main, so an explicit `using FileIO: save` would take
# precedence over the `save` that each example gets from `using Aquarium.CairoMakie`
# -- and the figures being written are Makie figures. Julia resolves globals in
# function bodies at call time, so leaving these unimported means the writers below
# pick up whatever `save`/`jldsave` the including script has in scope, which is the
# same binding the examples used before these wrappers existed.

#############################################################################################
## Artifact flags
#############################################################################################

# Examples compute and plot by default and write nothing at all. Saving is opt-in
# because the artifacts are expensive in different ways: animations need a video
# encoder and dominate runtime, and the data files are large. The code to produce
# them is a real part of what the examples demonstrate, so it stays -- it just
# stops being mandatory.
#
#   AQUARIUM_SAVE_DATA=true        .jld2 simulation data
#   AQUARIUM_SAVE_FIGURES=true     static figures
#   AQUARIUM_SAVE_ANIMATIONS=true  animations
#   AQUARIUM_SAVE_ALL=true         all three
#
# Flags are read once, here, so a single run is internally consistent.

_flag(name) = lowercase(strip(get(ENV, name, "false"))) in ("1", "true", "yes", "on")

const SAVE_ALL = _flag("AQUARIUM_SAVE_ALL")
const SAVE_DATA = SAVE_ALL || _flag("AQUARIUM_SAVE_DATA")
const SAVE_FIGURES = SAVE_ALL || _flag("AQUARIUM_SAVE_FIGURES")
const SAVE_ANIMATIONS = SAVE_ALL || _flag("AQUARIUM_SAVE_ANIMATIONS")

# Whether to open figures in a viewer. Defaults to whether the session is
# interactive: opening a window is what you want from a REPL or an editor cell,
# and is at best noise when the script is run headlessly -- on a cluster, over
# SSH, or in CI. That headless case is not hypothetical; see PR #2, whose commit
# "saved figs for headless VM" was working around exactly this.
#
#   AQUARIUM_DISPLAY=true|false   override the interactivity default
const DISPLAY_FIGURES = haskey(ENV, "AQUARIUM_DISPLAY") ? _flag("AQUARIUM_DISPLAY") : isinteractive()

#############################################################################################
## Output locations
#############################################################################################

"""
Root directory for everything the examples write.

Set `AQUARIUM_OUTPUT` to redirect it; otherwise output lands in `examples/output`,
which is ignored by version control.
"""
const OUTPUT_ROOT = get(ENV, "AQUARIUM_OUTPUT", joinpath(@__DIR__, "output"))

# These compute paths and deliberately do NOT create anything. Directories are
# made only when something is actually written, by the helpers below -- otherwise
# a default run would still litter the filesystem with empty directories.
visualization_dir(parts...) = joinpath(OUTPUT_ROOT, "visualization", parts...)
data_dir(parts...) = joinpath(OUTPUT_ROOT, "data", parts...)
data_file(parts...) = joinpath(OUTPUT_ROOT, "data", parts...)

# Announced after OUTPUT_ROOT exists, so the message can name the location.
if !(SAVE_DATA || SAVE_FIGURES || SAVE_ANIMATIONS)
    @info """
    Aquarium examples: artifact output is off, so this run writes nothing.
    Enable it with AQUARIUM_SAVE_ALL=true, or individually via
    AQUARIUM_SAVE_DATA / AQUARIUM_SAVE_FIGURES / AQUARIUM_SAVE_ANIMATIONS.
    Output would go to $(OUTPUT_ROOT) unless AQUARIUM_OUTPUT redirects it.
    Figures are $(DISPLAY_FIGURES ? "shown" : "not shown") in this session; override with AQUARIUM_DISPLAY."""
end

#############################################################################################
## Guarded writers
#############################################################################################

_prepare(path) = (mkpath(dirname(path)); path)

"""
    maybe_save(path, figure)

Save `figure` only when figure output is enabled. Returns `path` either way, so
callers can print or reuse it regardless.
"""
maybe_save(path, figure) = SAVE_FIGURES ? save(_prepare(path), figure) : path

"""
    maybe_jldsave(path; data...)

Write `data` only when data output is enabled. Returns `path` either way.
"""
maybe_jldsave(path; data...) = SAVE_DATA ? jldsave(_prepare(path); data...) : path

"""
    animate_if_enabled(f, args...; kwargs...)

Call animation function `f` only when animation output is enabled. Animations are
the slowest thing the examples do and need a video encoder, so they are off by
default.
"""
function animate_if_enabled(f, args...; kwargs...)
    SAVE_ANIMATIONS || return nothing
    for arg in args
        arg isa AbstractString && startswith(arg, OUTPUT_ROOT) && _prepare(arg)
    end
    return f(args...; kwargs...)
end

"""
    maybe_display(figure)

Open `figure` in a viewer only when display is enabled. Returns `figure` either
way, so it can be used inline.
"""
maybe_display(figure) = DISPLAY_FIGURES ? display(figure) : figure
