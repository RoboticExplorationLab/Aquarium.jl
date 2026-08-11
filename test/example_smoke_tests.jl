@testitem "Example scripts parse and use the shared output helpers" begin
    using Test

    # Nothing in the suite runs the example scripts -- they take minutes each and
    # need a display. That makes them the weakest-verified surface in the
    # repository: all thirteen can be edited wrongly while the suite stays green.
    #
    # This is a floor, not proof. It catches syntax damage and stale references to
    # the removed path constants, which is the failure mode a bulk edit actually
    # produces. It does not catch a script that parses fine and computes the wrong
    # thing.

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
        end
    end
end
