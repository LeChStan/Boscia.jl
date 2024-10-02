include("sparse_reg.jl")

################# Read ARGS passed #######################
mode = parse(String, ARGS[1])
dimension = parse(Int64, ARGS[2])
seed = parse(Int64, ARGS[3])
alternative = parse(String, ARGS[4])
decision_function = parse(String, ARGS[5])
iterations_until_stable =  parse(Int64, ARGS[6])
μ = parse(Float64, ARGS[7])

@show seed, dimension

################# Run example #######################
try 
    sparse_reg_shot(seed, dimension)
catch e 
    println(e)
    file = "shot_sparse_reg_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end