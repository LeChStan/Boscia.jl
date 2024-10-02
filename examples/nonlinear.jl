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




function build_examples(o, n,  seed)
    Random.seed!(seed)
    A = let
        A = randn(n, n)
        A' * A
    end
    
    @assert isposdef(A) == true
    
    y = Random.rand(Bool, n) * 0.6 .+ 0.3
    
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    for xi in x
        MOI.add_constraint(o, xi, MOI.ZeroOne())
        MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
        MOI.add_constraint(o, xi, MOI.LessThan(1.0))
    end
    lmo = Boscia.MathOptBLMO(o)

    function f(x)
        d = x - y
        return dot(d, A, d)
    end

    function grad!(storage, x)
        # storage = Ax
        mul!(storage, A, x)
        # storage = 2Ax - 2Ay
        return mul!(storage, A, y, -2, 2)
    end
    return f, grad!, lmo
end
   

function nonlinear_boscia(mode, dimension, seed, alternative, decision_function, iterations_until_stable, μ,  time_limit = 1800)
    o = SCIP.Optimizer()
    f, grad!, lmo = build_examples(o, dimension,  seed)
    branching_strategy, settings = build_branching_strategy(lmo, mode, alternative, decision_function, iterations_until_stable, μ)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
   # print(lmo.o)
    println("actual run")

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit, branching_strategy = branching_strategy)

    ### Define Parameters needed for documenting the results ###
    example_name = "nonlinear"
    file_name = "nonlinear_results"
    # Save result
    save_results(result, settings, μ, example_name, seed, dimension, file_name,false)
    return result
end

nonlinear_boscia("pseudocost", 20, 2, "most_infeasible", "product", 4, 1e-6,  600)