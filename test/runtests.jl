import Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.instantiate()

using TestItemRunner

@run_package_tests
