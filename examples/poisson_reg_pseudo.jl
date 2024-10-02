using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

# seed = 0x946d4b7835e92ffa takes 90 minutes to solve! -> not anymore
# seed = 0x946d4b7835e92ffa
# Random.seed!(seed)

# n = 30
# const ri = rand(n)
# const ai = rand(n)
# const Ωi = rand(Float64)
# const bi = sum(ai)
# Ai = randn(n, n)
# Ai = Ai' * Ai
# const Mi = (Ai + Ai') / 2
# @assert isposdef(Mi)


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

############ Decide which strategies to run #####################
strategies = Any[
    "MOST_INFEASIBLE", 
    #"Strong_Branching", 
    "hier"
]

for iterations_stable in Int64[5]
    for decision_function in [
        "product", 
        "weighted_sum"
        ]
        if decision_function == "product"
            μ = 1e-6
            push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
        else
            for μ in [0.7]
                push!(strategies, Dict(:iterations_stable => iterations_stable, :μ => μ, :decision_function => decision_function))
            end
        end
    end
end


############## Example sizes ######################

example_dimensions = [30, 35]

#seeds = rand(UInt64, 3)
seeds = [0x946d4b7835e92ffa]
modes = ["integer","mixed"]

############## Set Parameters for all runs ######################
verbose = true
print_iter = 100
time_limit = 600
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "portfolio_examples_a_d"
#################################################################



for seed in [1]
    f, grad!, n, ri, Ωi, Ai, Mi, ai, bi = build_function(seed, 10)
    #o = SCIP.Optimizer()
    o = HiGHS.Optimizer()
    lmo, _ = build_optimizer(o, "integer", n, ai, bi)
    # println(o)
    println("presolve")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10, use_postsolve=false) 
end
println("actual solve")

for seed in seeds
    for dim in example_dimensions
        for mode in modes
            example_name = string("portfolio_dim_", dim, "_mode_", mode)
            for branching_strategy in strategies
                f, grad!, n, ri, Ωi, Ai, Mi, ai, bi = build_function(seed, dim)
                o = SCIP.Optimizer()
                #o = HiGHS.Optimizer()
                lmo, _ = build_optimizer(o, mode, n, ai, bi)
                if branching_strategy == "Strong_Branching"
                    blmo = Boscia.MathOptBLMO(HiGHS.Optimizer())
                    branching_strategy = Boscia.PartialStrongBranching(10, 1e-3, blmo)
                    MOI.set(branching_strategy.bounded_lmo.o, MOI.Silent(), true)
                    x, _, result =
                        Boscia.solve(
                            f, 
                            grad!, 
                            lmo,  
                            branching_strategy=branching_strategy,verbose=verbose,
                            print_iter=print_iter, 
                            time_limit=time_limit,
                            rel_dual_gap=rel_dual_gap
                        )
                    settings = "Strong_Branching"
                    #Boscia.save_results(result, settings, example_name, seed, file_name, false) 
                
                elseif branching_strategy == "MOST_INFEASIBLE"
                    x, _, result = Boscia.solve(
                        f, 
                        grad!, 
                        lmo, 
                        verbose=verbose, 
                        print_iter=print_iter, 
                        time_limit=time_limit,
                        rel_dual_gap=rel_dual_gap
                        )
                    settings = "MOST_INFEASIBLE"
                    #Boscia.save_results(result, settings, example_name, seed, file_name, false)
                elseif branching_strategy == "hier"
                    branching_strategy = Boscia.HIERARCHY_PSEUDO_COST(4, false, lmo, 0.7, "weighted_sum") 
                    x, _, result = Boscia.solve(f, grad!, lmo, branching_strategy = Boscia.HIERARCHY_PSEUDO_COST(4, false, lmo, 0.7, "weighted_sum"),verbose=true)
                    #settings = "HIERARCHY_PSEUDO_COST_" * branching_strategy.decision_function * "_" * string(branching_strategy.gradient_influence) * "_" * string(branching_strategy.iterations_until_stable) * "_μ=" * string(branching_strategy.μ)
                    #Boscia.save_results(result, settings, example_name, seed, file_name, false)
                else
                    iterations_stable = branching_strategy[:iterations_stable]
                    decision_function = branching_strategy[:decision_function]
                    μ = branching_strategy[:μ]
                    x, _, result = Boscia.solve(
                        f, 
                        grad!, 
                        lmo,
                        branching_strategy=Boscia.PSEUDO_COST(iterations_stable,false, lmo, μ, decision_function),
                        verbose=verbose, 
                        print_iter=print_iter, 
                        time_limit=time_limit,
                        rel_dual_gap=rel_dual_gap
                        )
                #         settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
                #     Boscia.save_results(result, settings, example_name, seed, file_name, false)
                end
            end
        end
    end
end