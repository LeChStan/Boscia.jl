using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS
# using Statistics
using LinearAlgebra
using Distributions
using CSV
using DataFrames
import MathOptInterface
const MOI = MathOptInterface
include("example_utilities.jl")


# The example from  "Optimizing a low-dimensional convex function over a high-dimensional cube"
# by Christoph Hunkenschröder, Sebastian Pokutta, Robert Weismantel
# https://arxiv.org/abs/2204.05266. 

function low_dim_high_dim(o, m, n, seed; alpha=0.00)
    Random.seed!(seed)
    refpoint = 0.5 * ones(n) + Random.rand(n) * alpha * 1 / n
    W = rand(m, n)
    Ws = transpose(W) * W
    function f(x)
        return 0.5 * (dot(x, Ws, x) - dot(refpoint, Ws, x) - dot(x, Ws, refpoint))
    end
    
    function grad!(storage, x)
        return mul!(storage, Ws, (x - refpoint))
    end

    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
        MOI.add_constraint(o, xi, MOI.ZeroOne())
    end
    lmo = Boscia.MathOptBLMO(o)
    return lmo, f, grad!
end 

function low_dim_high_dim_boscia(mode, low_dimension, high_dimension, seed, alternative, decision_function, iterations_until_stable, μ,  time_limit = 1800)
    o = SCIP.Optimizer()
    lmo, f, grad! = low_dim_high_dim(o, high_dimension, low_dimension, seed; alpha=0.00)
   
    branching_strategy, settings = build_branching_strategy(lmo, mode, alternative, decision_function, iterations_until_stable, μ)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
   # print(lmo.o)
    println("actual run")

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit, branching_strategy = branching_strategy)

    ### Define Parameters needed for documenting the results ###
    example_name = "low_dim_high_dim_" * string(high_dimension)
    file_name = "low_dim_high_dim_results"
    # Save result
    save_results(result, settings, μ, example_name, seed, dimension, file_name,false)
    return result
end

#low_dim_high_dim_boscia("strong_branching", 500, 10, 1, "na", "na", 1, 1e-6, 180)