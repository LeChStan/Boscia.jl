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

# Integer sparse regression

# min norm(y-A x)² 
# s.t. 0 <= x_i <= r
# ∑ x_i <= k 
# x_i ∈ Z for i = 1,..,n

# There A represents the collection of data points and 
# is a very tall matrix, i.e. number of rows = m >> number of columns = n.
# y - is the vector of results.
# r - controls how often we have to maximal split on a index.
# k - is the sparsity parameter. We only want a few non zero entries.
function int_sparse_regression(o, n, m, l, k, seed)
    Random.seed!(seed)
    sol_x = rand(1:l, n)
    for _ in 1:(n-k)
        sol_x[rand(1:n)] = 0
    end
    D = rand(m, n)
    y_d = D * sol_x

    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, n)
    z = MOI.add_variables(o, n)
    for i in 1:n
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(1.0 * l))
        MOI.add_constraint(o, x[i], MOI.Integer())

        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())

        MOI.add_constraint(o, 1.0 * x[i] - 1.0 * l * z[i], MOI.LessThan(0.0))
    end
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(zeros(n),x), sum(Float64.(iszero.(x)))), MOI.GreaterThan(1.0*(n-k)))
    # MOI.add_constraint(o, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n),z), 0.0), MOI.GreaterThan(1.0*k))
    lmo = Boscia.MathOptBLMO(o)

    function f(x)
        xv = @view(x[1:n])
        return 1 / 2 * sum(abs2, y_d - D * xv)  #+ lambda_2*FrankWolfe.norm(x)^2 + lambda_0*sum(x[p+1:2p])
    end

    function grad!(storage, x)
        storage .= 0
        @view(storage[1:n]) .= transpose(D) * (D * @view(x[1:n]) - y_d)
        return storage
    end

    return lmo, f, grad!
end

function int_sparse_reg_boscia(mode, dimension, seed, alternative, decision_function, iterations_until_stable, μ,  time_limit = 1800)
    
    o = SCIP.Optimizer()

    n = dimension
    m = 3* dimension
    l = ceil(dimension/ 2)
    k = l - 1
    lmo, f, grad! = int_sparse_regression(o, n, m, l, k, seed)

    branching_strategy, settings = build_branching_strategy(lmo, mode, alternative, decision_function, iterations_until_stable, μ)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
   # print(lmo.o)
    println("actual run")

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit, branching_strategy = branching_strategy)

    ### Define Parameters needed for documenting the results ###
    example_name = "int_sparse_reg"
    file_name = "int_sparse_reg_results"
    # Save result
    save_results(result, settings, μ, example_name, seed, dimension, file_name,false)
    return result
end

#int_sparse_reg_boscia("strong_branching", 10, 1, "na", "na", 1, 1e-6, 180)