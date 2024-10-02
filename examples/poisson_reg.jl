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
# Poisson sparse regression

# min_{w, b, z} ∑_i exp(w x_i + b) - y_i (w x_i + b) + α norm(w)^2
# s.t. -N z_i <= w_i <= N z_i
# b ∈ [-N, N]
# ∑ z_i <= k 
# z_i ∈ {0,1} for i = 1,..,p

# y_i    - data points, poisson distributed 
# X_i, b - coefficient for the linear estimation of the expected value of y_i
# w_i    - continuous variables
# z_i    - binary variables s.t. z_i = 0 => w_i = 0
# k      - max number of non zero entries in w

# In a poisson regression, we want to model count data.
# It is assumed that y_i is poisson distributed and that the log 
# of its expected value can be computed linearly.

function build_function(seed, n; Ns=0.0, use_scale=false)
    Random.seed!(seed)
    p = n

    # underlying true weights
    ws = rand(Float64, p)
    # set 50 entries to 0
    for _ in 1:20
        ws[rand(1:p)] = 0
    end
    bs = rand(Float64)
    Xs = randn(Float64, n, p)
    for j in 1:p 
        Xs[:,j] ./= (maximum(Xs[:,j]) - minimum(Xs[:,j]))
        if Ns == 10.0
            Xs[:,j] .*= 0.1
        end
    end
    ys = map(1:n) do idx
        a = dot(Xs[idx, :], ws) + bs
        return rand(Distributions.Poisson(exp(a)))
    end

    α = 1.3
    scale = exp(n/2)
    function f(θ)
        #θ = BigFloat.(θ)
        w = @view(θ[1:p])
        b = θ[end]
        s = sum(1:n) do i
            a = dot(w, Xs[:, i]) + b
            return 1 / n * (exp(a) - ys[i] * a)
        end
        if use_scale
            return 1/scale * (s + α * norm(w)^2)
        end
        return s + α * norm(w)^2
    end
    function grad!(storage, θ)
        #θ = BigFloat.(θ)
        w = @view(θ[1:p])
        b = θ[end]
        storage[1:p] .= 2α .* w
        storage[p+1:2p] .= 0
        storage[end] = 0
        for i in 1:n
            xi = @view(Xs[:, i])
            a = dot(w, xi) + b
            storage[1:p] .+= 1 / n * xi * exp(a)
            storage[1:p] .-= 1 / n * ys[i] * xi
            storage[end] += 1 / n * (exp(a) - ys[i])
        end
        if use_scale
            storage .*= 1/scale
        end
        return storage
    end
    # @show bs, Xs, ys, ws

    return f, grad!, p, α, bs, Xs, ys, ws
end

function build_optimizer(o, p, k, Ns)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    w = MOI.add_variables(o, p)
    z = MOI.add_variables(o, p)
    b = MOI.add_variable(o)
    # z_i ∈ {0,1} for i = 1,..,p
    for i in 1:p
        MOI.add_constraint(o, z[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, z[i], MOI.LessThan(1.0))
        MOI.add_constraint(o, z[i], MOI.ZeroOne())
    end
    for i in 1:p
        # s.t. -N z_i <= w_i <= N z_i
        MOI.add_constraint(o, Ns * z[i] + w[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, -Ns * z[i] + w[i], MOI.LessThan(0.0))
        # Indicator: z[i] = 1 => -N <= w[i] <= N
        #=gl = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, w[i])),],
            [0.0, 0.0], )
        gg = MOI.VectorAffineFunction(
            [   MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, z[i])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1.0, w[i])),],
            [0.0, 0.0], )
        MOI.add_constraint(o, gl, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(Ns)))
        MOI.add_constraint(o, gg, MOI.Indicator{MOI.ACTIVATE_ON_ONE}(MOI.LessThan(-Ns))) =#
    end
    # ∑ z_i <= k 
    MOI.add_constraint(o, sum(z, init=0.0), MOI.LessThan(1.0 * k))
    MOI.add_constraint(o, sum(z, init=0.0), MOI.GreaterThan(1.0))
    # b ∈ [-N, N]
    MOI.add_constraint(o, b, MOI.LessThan(Ns))
    MOI.add_constraint(o, b, MOI.GreaterThan(-Ns))
    lmo = Boscia.MathOptBLMO(o)
    return lmo, (w,z,b)
end


function poisson_reg_boscia(mode, dimension, seed, alternative, decision_function, iterations_until_stable, μ,  time_limit = 1800)
    Ns=0.1
    f, grad!, p, α, bs, Xs, ys, ws = build_function(seed, dimension; Ns=Ns)
    k = dimension/2
    o = SCIP.Optimizer()
    lmo, _ = build_optimizer(o, p, k, Ns)
    branching_strategy, settings = build_branching_strategy(lmo, mode, alternative, decision_function, iterations_until_stable, μ)
    # println(o)
    println("precompile")
    Boscia.solve(f, grad!, lmo, verbose=false, time_limit=10)
   # print(lmo.o)
    println("actual run")

    x, _, result = Boscia.solve(f, grad!, lmo, verbose=true, time_limit=time_limit, branching_strategy = branching_strategy)

    ### Define Parameters needed for documenting the results ###
    example_name = "poisson_regression"
    file_name = "poisson_reg_results"
    # Save result
    save_results(result, settings, μ, example_name, seed, dimension, file_name,false)
    return result
end

#poisson_reg_boscia("strong_branching", 20, 2, "most_infeasible", "product", 4, 1e-6,  600)