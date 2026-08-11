@testitem "Example scripts parse and use the shared output helpers" begin
    using Test

    # Nothing in the suite runs the example scripts -- they take minutes each and
    # need a display. That makes them the weakest-verified surface in the
    # repository: all thirteen can be edited wrongly while the suite stays green.
    #
    # These checks are a floor, not proof. They catch syntax damage, stale
    # references to removed constants, and artifact calls that escaped a guard --
    # the failure modes a bulk edit actually produces. They do not catch a script
    # that parses fine and computes the wrong thing.

    examples_root = joinpath(pkgdir(Aquarium), "examples")
    scripts = String[]
    for (dir, _, files) in walkdir(examples_root), file in files
        endswith(file, ".jl") && file != "common.jl" && push!(scripts, joinpath(dir, file))
    end

    @test length(scripts) == 13

    for script in scripts
        source = read(script, String)
        rel = relpath(script, examples_root)

        @testset "$rel" begin
            # Parses as valid Julia.
            parsed = Meta.parseall(source; filename = script)
            @test !any(x -> x isa Expr && x.head === :error, parsed.args)

            # Uses the shared setup rather than its own environment handling.
            @test occursin("common.jl", source)
            @test !occursin(r"^\s*Pkg\."m, source)

            # References no constant the library no longer defines.
            for removed in ("Aquarium.VIS_DIR", "Aquarium.DATA_DIR",
                            "Aquarium.EXAMPLES_DIR", "Aquarium.TEST_DIR")
                @test !occursin(removed, source)
            end

            # Every artifact-producing call goes through a guard, so a default run
            # writes nothing. A missed call site is exactly what a bulk edit leaves
            # behind, and it would silently reintroduce unconditional file output.
            @test !occursin(r"(?<!maybe_)\bsave\(", source)
            @test !occursin(r"(?<!maybe_)\bjldsave\(", source)

            animate_calls = [m.match for m in eachmatch(r"\banimate_[a-z_]+\(", source)]
            unguarded = filter(c -> c != "animate_if_enabled(", animate_calls)
            @test isempty(unguarded)

            # No example reads back data it wrote in the same run. That round trip
            # only worked because saving was unconditional.
            @test !occursin(r"\bload\(", source)

            # Figures are opened through the display guard, so a headless run --
            # cluster, CI, over SSH -- does not try to open a viewer.
            @test !occursin(r"(?<!maybe_)\bdisplay\(", source)
        end
    end
end
