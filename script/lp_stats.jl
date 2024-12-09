include("../src/SimplePDLP.jl")
using ArgParse
using LinearAlgebra

function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()
  
    ArgParse.@add_arg_table! arg_parse begin
      "--problem_folder"
      help = "The directory for instances."
      arg_type = String
      # required = true
      default = "./data"
  
      "--dataset"
      help = "The LP dataset."
      arg_type = String
      default = "netlib"
  
      "--output_directory"
      help = "The directory for output files."
      arg_type = String
      # required = true
      default = "./assets"
  
    end
  
    return ArgParse.parse_args(arg_parse)
  end

function opt_primal_dual_solution(
    problem::LinearProgrammingProblem,
)
    m, n = size(problem.constraint_matrix)
    c = problem.objective_vector
    A = problem.constraint_matrix
    b = problem.right_hand_side
    lb = problem.variable_lower_bound
    ub = problem.variable_upper_bound
    num = problem.num_equalities

    x = Variable(n)
    y = Variable(m)

    primal = minimize(c' * x, 
        [A[1:num,:] * x == b[1:num], 
        A[num+1:end,:] * x >= b[num+1:end],
        lb .<= x, x .<= ub])
    dual = maximize(b' * y, [A' * y + x == c, y .>= 0])

    solve!(primal)
    solve!(dual)

    return primal, dual
end

function main()
    parsed_args = parse_command_line()
    problem_folder = parsed_args["problem_folder"]
    dataset = parsed_args["dataset"]
    output_directory = parsed_args["output_directory"]

    problem_folder = joinpath(problem_folder, dataset)

    all_instances = readdir(problem_folder)
    all_instances = [replace(basename(instance), ".mps.gz" => "") for instance in all_instances if endswith(instance, ".mps.gz")]
    len = length(all_instances)

    csv_file = joinpath(output_directory, "$(dataset)_norm.csv")

    open(csv_file, "w") do io
        println(io, "name, n ,m, nnz, density, norm_before, norm_after")
    
        for i in 1:len
            problem_name = all_instances[i]
            instance_path = joinpath(problem_folder, "$(problem_name).mps.gz")
            
            try 
                lp = qps_reader_to_standard_form(instance_path)
                m, n = size(lp.constraint_matrix)
                if n > 5000 && m > 5000
                    continue
                end
                nnz = length(lp.constraint_matrix.nzval)
                density = nnz / (m * n)
                norm_before = norm(lp.constraint_matrix)
                scaled_problem = rescale_problem(10,true,4,lp)
                scaled_lp = scaled_problem.scaled_lp
                norm_after = norm(scaled_lp.constraint_matrix)
                println(io, "$(problem_name),$(n),$(m),$(nnz),$(density),$(norm_before),$(norm_after)")
            catch e
                println("Error in read $(i), $(problem_name)", e)
                continue
            end
        end
    end # open
end # main

main()