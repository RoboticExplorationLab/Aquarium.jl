@testitem "Loading Aquarium writes nothing to the user's home directory" begin
    using Aquarium

    # The naive form of this test -- assert ~/aquarium does not exist -- is
    # useless in both directions. It passes vacuously on a machine that never ran
    # the old code, and fails spuriously on a developer machine where an earlier
    # run already created the directory. Neither outcome says anything about the
    # current code.
    #
    # Load the package in a subprocess with HOME pointed at an empty temporary
    # directory, then assert that directory is still empty. JULIA_DEPOT_PATH is
    # passed through explicitly: the depot is derived from HOME by default, so
    # without this the subprocess would populate the fake home with a .julia
    # directory and the test would fail for reasons unrelated to Aquarium.
    mktempdir() do fake_home
        env = copy(ENV)
        env["HOME"] = fake_home
        env["JULIA_DEPOT_PATH"] = join(DEPOT_PATH, Sys.iswindows() ? ';' : ':')

        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e "using Aquarium"`
        @test success(run(setenv(cmd, env); wait = true))

        @test isempty(readdir(fake_home))
    end
end

@testitem "Aquarium exposes no filesystem-path constants" begin
    using Aquarium

    # These four constants were removed along with the mkpath calls that ran at
    # module scope. Two of them pointed into the package's own source tree, which
    # is read-only for any registry install -- see docs/adr/0003. Asserting their
    # absence keeps a future contributor from reintroducing the convenience.
    for name in (:VIS_DIR, :DATA_DIR, :EXAMPLES_DIR, :TEST_DIR)
        @test !isdefined(Aquarium, name)
    end
end
