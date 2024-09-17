using Statistics
using Boscia
using FrankWolfe
using Random
using LinearAlgebra
using SCIP
using HiGHS
import Bonobo
import MathOptInterface
const MOI = MathOptInterface
using Dates
using Printf
using Test

# Sparse regression

# Constant parameters for the sparse regression
# min norm(y-A β)² + λ_0 ∑ z_i + λ_2 ||β||²
# s.t. -Mz_i <= β_i <= Mz_i
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p 

# A - matrix of observations.
# y - vector of results.
# We want to match Aβ as closely as possible to y 
# while having relative few non zero entries in β.
# Each continuous variable β_i is assigned a binary z_i,
# z_i = 0 => β_i = 0

# example_name = "sparse_regression"
# n0 = 10;
# p = 5 * n0;
# k = ceil(n0 / 5);
# const lambda_0 = rand(Float64);
# const lambda_2 = 10.0 * rand(Float64);
# const A = rand(Float64, n0, p)
# const y = rand(Float64, n0)
# const M = 2 * var(A)

function build_function(seed, n)
    Random.seed!(seed)
    p = 5 * n;
    k = ceil(n / 5);
    lambda_0 = rand(Float64);
    lambda_2 = 10.0 * rand(Float64);
    A = rand(Float64, n, p)
    y = rand(Float64, n)
    M = 2 * var(A)
    # @show A, y, M, lambda_0, lambda_2

    function f(x)
        xv = @view(x[1:p])
        return norm(y - A * xv)^2 + lambda_0 * sum(x[p+1:2p]) + lambda_2 * norm(xv)^2
    end

    function grad!(storage, x)
        storage[1:p] .= 2 * (transpose(A) * A * x[1:p] - transpose(A) * y + lambda_2 * x[1:p])
        storage[p+1:2p] .= lambda_0
        return storage
    end

    return f, grad!, p, k, M, A, y, lambda_0, lambda_2
end

function build_optimizer(o, p, k, M)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, 2p)
    for i in p+1:2p
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, x[i], MOI.ZeroOne()) # or MOI.Integer()
    end
    for i in 1:p
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, M], [x[i], x[i+p]]), 0.0),
            MOI.GreaterThan(0.0),
        )
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -M], [x[i], x[i+p]]), 0.0),
            MOI.LessThan(0.0),
        )
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(p), x[p+1:2p]), 0.0),
        MOI.LessThan(k),
    )
    lmo = Boscia.MathOptBLMO(o)
    return lmo, x
end






############ Decide which strategies to run #####################
strategies = Any[
    "MOST_INFEASIBLE", "Strong_Branching"
]

for iterations_stable in Int64[5,10]
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

no_choices = Int[10,
#40
]

seeds = rand(UInt64, 3)

############## Set Parameters for all runs ######################
verbose = true
print_iter = 100
time_limit = 600
rel_dual_gap=1e-2
# Set parameters for saving results
file_name = "sparse_reg_examples_a_c"
#example_name = string("int_sparse_reg_n_", n, "_m_", m, "_l_",l, "_k_", k)
f, grad!, p, k, M, A, y, lambda_0, lambda_2 = build_function(1, 10)
o = SCIP.Optimizer()
lmo, _ = build_optimizer(o, p, k, M)
# println(o)
println("precompile")
Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
# print(lmo.o)
println("actual run")#################################################################


for seed in seeds
    for dim in no_choices
        n0 = dim       
        for branching_strategy in strategies
            f, grad!, p, k, M, A, y, lambda_0, lambda_2 = build_function(seed, n0)
            o = SCIP.Optimizer()
            example_name = string("sparse_reg_n0_", n0, "_p_", p, "_k_",k)
            lmo, _ = build_optimizer(o, p, k, M)
            if branching_strategy == "Strong_Branching"
                #blmo = Boscia.MathOptBLMO(SCIP.Optimizer())
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
                Boscia.save_results(result, settings, example_name, seed, file_name, false) 
            
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
                Boscia.save_results(result, settings, example_name, seed, file_name, false) 
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
                    settings = "PSEUDO_COST_" * decision_function * "_" * string(iterations_stable) * "_μ=" * string(μ)
                Boscia.save_results(result, settings, example_name, seed, file_name, false)
            end
        end
    end
end

