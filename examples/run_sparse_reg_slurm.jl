modes = ["most_infeasible", "strong_branching", "pseudocost", 
         "hierarchy", "largest_gradient", "largest_most_infeasible_gradient"] 

for mode in modes
    @show mode
    for dimension in 15:30
        for seed in 1:10
            if mode == "pseudocost" || mode == "hierarchy"
                depths = [1, 2, 5, 10]
            else 
                depths = [1]
            end

            if mode == "hierarchy"
                alternatives = ["default", "largest_most_infeasible_gradient", "largest_gradient"]
            end

            @show seed, dimension
            run(`sbatch batch_sparse_reg.sh $mode $dimension $seed`)
        end 
    end
end