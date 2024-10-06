include("int_sparse_reg.jl")

################# Read ARGS passed #######################
mode = parse(String, ARGS[1])
dimension = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
alternative = parse(String, ARGS[4])
decision_function = parse(String, ARGS[5])
iterations_until_stable =  parse(Int64, ARGS[6])
μ = parse(Float64, ARGS[7])
time_limit = 1800
@show seed, dimension

################# Run example #######################
try 
    int_sparse_reg_boscia(mode, dimension, seed, alternative, decision_function, iterations_until_stable, μ,  time_limit)
catch e 
    println(e)
    file = "int_sparse_reg_" * str(seed) * "_" * str(dimension) * "_" * mode 
    open(file * ".txt","a") do io
        println(io, e)
    end
end