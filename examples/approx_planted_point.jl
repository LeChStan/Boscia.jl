using Boscia
using FrankWolfe
using Bonobo
using Random
using SCIP
using LinearAlgebra
using Distributions
using DataFrames, CSV
using HiGHS
import MathOptInterface
const MOI = MathOptInterface
include("example_utilities.jl")


function approx_planted_point_integer(o, n, seed)
    Random.seed!(seed)
    diffi = Random.rand(Bool, n) * 0.6 .+ 0.3
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end
    # using SCIP

    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
    end
    lmo = Boscia.MathOptBLMO(o)
    return lmo, f, grad!
end

function approx_planted_point_mixed(o, n, seed)
    Random.seed!(seed)
    diffi = Random.rand(Bool, n) * 0.6 .+ 0.3
    function f(x)
        return 0.5 * sum((x[i] - diffi[i])^2 for i in eachindex(x))
    end
    function grad!(storage, x)
        @. storage = x - diffi
    end
    int_vars = unique!(rand(collect(1:n), Int(floor(n / 2))))
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        if xi.value in int_vars
            MOI.add_constraint(o, xi, MOI.ZeroOne()) # or MOI.Integer()
        end
    end
    lmo = Boscia.MathOptBLMO(o)

    return lmo, f, grad!
end 


function approx_planted_boscia(mode, dimension, seed, alternative, decision_function, iterations_until_stable, μ, set, time_limit = 1800)

    o = SCIP.Optimizer()
    #o = HiGHS.Optimizer()
    if set == "mixed"
        lmo, f, grad! = approx_planted_point_mixed(o, n, seed)
    else 
        lmo, f, grad! = approx_planted_point_integer(o, n, seed)
    end
    branching_strategy, settings = build_branching_strategy(lmo, mode, alternative, decision_function, iterations_until_stable, μ)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
   # print(lmo.o)
    println("actual run")

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit, branching_strategy = branching_strategy)

    ### Define Parameters needed for documenting the results ###
    example_name = "approx_planted_" * set 
    file_name = "approx_planted_" * set * "_results"
    # Save result
    save_results(result, settings, μ, example_name, seed, dimension, file_name,false)
    return result
end