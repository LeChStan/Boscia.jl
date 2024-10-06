modes = ["most_infeasible", "strong_branching", "pseudocost", 
         "hierarchy", "largest_gradient", "largest_most_infeasible_gradient"] 

for mode in modes
    @show mode
    for dimension in 20:35
        for seed in 1:10
            if mode == "pseudocost" || mode == "hierarchy"
                depths = [1, 2, 5, 10]
            else 
                depths = [1]
            end

            if mode == "hierarchy" || mode == "pseudocost"
                #Strategies to use when pseudos not stable
                alternatives = ["most_infeasible", "largest_most_infeasible_gradient", "largest_gradient"]
                #Strategy to use when pseudos stable
                decision_functions = ["weighted_sum", "product"]
            else 
                alternatives = ["na"]
                decision_functions = ["na"]
            end
            for depth in depths
                for alternative in alternatives
                    for decision_function in decision_functions
                        if decision_function == "weighted_sum"
                            weights = [0.1, 0.3, 0.5, 0.7, 0.9]
                        else
                            weights = [1e-6]
                        end
                        for weight in weights
                            @show seed, dimension
                            run(`sbatch batch_int_sparse_reg.sh $mode $dimension $seed $alternative $decision_function $depth $weight`)
                        end
                    end
                end
            end
        end 
    end
end