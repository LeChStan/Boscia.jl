include("sparse_reg.jl")
include("save_result.jl")

seed = parse(Int64, ARGS[1])
dimension = parse(Int64, ARGS[2])
@show seed, dimension

try 
    sparse_reg_shot(seed, dimension)
catch e 
    println(e)
    file = "shot_sparse_reg_" * str(seed) * "_" * str(dimension)    
    open(file * ".txt","a") do io
        println(io, e)
    end
end