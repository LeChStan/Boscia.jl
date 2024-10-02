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

function build_function(seed, dimension)
    @show seed
    Random.seed!(seed)
    n = dimension
    ri = rand(n)
    Ωi = rand()
    Ai = randn(n, n)
    Ai = Ai' * Ai
    Mi = (Ai + Ai') / 2
    @assert isposdef(Mi)

    ai = rand(dimension)
    bi = sum(ai)

    function f(x)
        return 1 / 2 * Ωi * dot(x, Mi, x) - dot(ri, x)
    end
    function grad!(storage, x)
        mul!(storage, Mi, x, Ωi, 0)
        storage .-= ri
        return storage
    end
    return f, grad!, n, ri, Ωi, Ai, Mi, ai, bi
end

function build_optimizer(o, mode, n, ai, bi)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    println("build optimizer")
    # @show ai, bi

    # MOI.set(o, MOI.TimeLimitSec(), limit)
    x = MOI.add_variables(o, n)
    
    # integer set
    if mode == "integer"
        I = collect(1:n)
    elseif mode == "mixed"
        I = 1:(n÷2)
    end

    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if i in I
            MOI.add_constraint(o, x[i], MOI.Integer())
        end
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ai, x), 0.0),
        MOI.LessThan(bi),
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.GreaterThan(1.0),
    )
    lmo = Boscia.MathOptBLMO(o)
    return lmo, x
end

function portfolio_boscia(mode, dimension, seed, alternative, decision_function, iterations_until_stable, μ, set, time_limit = 1800)
    f, grad!, n, ri, Ωi, Ai, Mi, ai, bi = build_function(seed, dimension)
    #o = SCIP.Optimizer()
    o = HiGHS.Optimizer()
    lmo, _ = build_optimizer(o, set, n, ai, bi)
    branching_strategy, settings = build_branching_strategy(lmo, mode, alternative, decision_function, iterations_until_stable, μ)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
   # print(lmo.o)
    println("actual run")

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit, branching_strategy = branching_strategy)

    ### Define Parameters needed for documenting the results ###
    example_name = "portfolio_" * set 
    file_name = "portfolio_" * set * "_results"
    # Save result
    save_results(result, settings, μ, example_name, seed, dimension, file_name,false)
    return result
end

#portfolio_boscia("strong_branching", 20, 2, "most_infeasible", "product", 4, 1e-6, "integer", 600)
#portfolio_boscia("strong_branching", 20, 2, "most_infeasible", "product", 4, 1e-6, "mixed", 600)