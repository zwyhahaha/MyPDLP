"""
A PrimalDualOptimizerParameters struct specifies the parameters for solving the saddle point formulation of an problem using primal-dual hybrid gradient. It solves a problem of the form 
    minimize objective_vector' * x
    s.t. constraint_matrix[1:num_equalities, :] * x = right_hand_side[1:num_equalities]
         constraint_matrix[(num_equalities + 1):end, :] * x >= right_hand_side[(num_equalities + 1):end, :]
         variable_lower_bound <= x <= variable_upper_bound

We use notation from Chambolle and Pock, "On the ergodic convergence rates of a
first-order primal-dual algorithm"
(http://www.optimization-online.org/DB_FILE/2014/09/4532.pdf).
That paper doesn't explicitly use the terminology "primal-dual hybrid gradient"
but their Theorem 1 is analyzing PDHG. In this file "Theorem 1" without further
reference refers to that paper.

Our problem is equivalent to the saddle point problem:
    min_x max_y L(x, y)
where
    L(x, y) = y' K x + g(x) - h*(y)
    K = -constraint_matrix
    g(x) = objective_vector' x if variable_lower_bound <= x <= variable_upper_bound, otherwise infinity
    h*(y) = -right_hand_side' y if y[(num_equalities + 1):end] >= 0, otherwise infinity

Note that the places where g(x) and h*(y) are infinite effectively limits the domain of the min and max. Therefore there's no infinity in the code. We use the same step size for both primal and dual (tau and sigma in Chambolle and
Pock) as:
    primal_step_size = step_size
    dual_step_size = step_size.

The algoritm converges if
    primal_stepsize * dual_stepsize * norm(contraint_matrix)^2 < 1.
"""
mutable struct PrimalDualOptimizerParameters
  """
  Constant step size used in the algorithm. If nothing is specified, the solver
  computes a provably correct step size.
  """
  step_size::Union{Float64,Nothing}
  """
  If true the LP instance is rescaled.
  """
  rescale_flag::Bool
  """
  Records iteration stats to csv every record_every iterations.
  During these iteration restarts may also be performed.
  """
  record_every::Int64
  """
  If verbose is true then prints iteration stats every print_every time that
  iteration information is recorded.
  """
  print_every::Int64
  """
  If true a line of debugging info is printed every printing_every
  iterations.
  """
  verbosity::Bool
  """
  Number of loop iterations to run. Must be postive.
  """
  iteration_limit::Int64
  """
  Tolerance for KKT error. Must be postive.
  """
  kkt_tolerance::Float64
  """
  Initial primal solution
  """
  initial_primal_solution::Vector{Float64}
  """
  Initial dual solution
  """
  initial_dual_solution::Vector{Float64}
  """
  whether to use online scaling
  """
  online_scaling::Bool
  """
  learning rate for the online scaling
  """
  learning_rate::Float64
end


@enum SolutionStatus STATUS_OPTIMAL STATUS_ITERATION_LIMIT
"""
Output of the solver.
"""
struct PrimalDualOutput
  primal_solution::Vector{Float64}
  dual_solution::Vector{Float64}
  iteration_stats::DataFrames.DataFrame
  time::Float64
  iteration::Int64
  primal_size::Int64
  dual_size::Int64
  status::SolutionStatus
end

"""
Given primal-dual solution pair (x,y)
compute dual_residual = c - A'*y
"""
function compute_dual_residual(
  dual_iterate::Vector{Float64},
  problem::LinearProgrammingProblem,
)
  @inbounds return problem.objective_vector - problem.constraint_matrix' * dual_iterate
end

"""
Given primal-dual solution pair (x,y)
compute primal_residual = b - A*x
"""
function compute_primal_residual(
  primal_iterate::Vector{Float64},
  problem::LinearProgrammingProblem,
)
  @inbounds return problem.right_hand_side .- problem.constraint_matrix * primal_iterate
end

function projection_primal(
    primal_iterate::Vector{Float64},
    problem::LinearProgrammingProblem,)

    lower_bound, upper_bound = problem.variable_lower_bound, problem.variable_upper_bound
    for idx in 1:length(primal_iterate)
      primal_iterate[idx] = min(
        upper_bound[idx],
        max(lower_bound[idx], primal_iterate[idx]),
      )
    end
	return primal_iterate
end

function projection_dual(
    dual_iterate::Vector{Float64},
    problem::LinearProgrammingProblem)

    for idx in (problem.num_equalities+1):length(dual_iterate)
      dual_iterate[idx] = max(dual_iterate[idx], 0.0)
    end
	return dual_iterate
end

function take_pdhg_step(
  problem::LinearProgrammingProblem,
  current_primal_solution::Vector{Float64},
  current_dual_solution::Vector{Float64},
  current_primal_residual::Vector{Float64},
  current_dual_residual::Vector{Float64},
  primal_step_size::Union{Float64, Vector{Float64}},
  dual_step_size::Union{Float64, Vector{Float64}}
)
  @inbounds next_primal = projection_primal(current_primal_solution .- primal_step_size .* current_dual_residual, problem)
  @inbounds next_primal_residual = compute_primal_residual(next_primal, problem)
  @inbounds next_dual = projection_dual(current_dual_solution .- dual_step_size .* (current_primal_residual .- 2 .* next_primal_residual), problem)
  @inbounds next_dual_residual = compute_dual_residual(next_dual, problem)
  return next_primal, next_dual, next_primal_residual, next_dual_residual
end

function step_size_update!(
  primal_step_size::Vector{Float64},
  dual_step_size::Vector{Float64},
  learning_rate::Float64,
  primal_step_size_v::Vector{Float64},
  dual_step_size_v::Vector{Float64},
  current_primal_residual::Vector{Float64},
  current_dual_residual::Vector{Float64},
  next_primal_residual::Vector{Float64},
  next_dual_residual::Vector{Float64},
)
    primal_step_size_gradient = - next_dual_residual .* current_dual_residual / (norm(current_dual_residual)^2 + eps())
    dual_step_size_gradient = - next_primal_residual .* (current_primal_residual-2*next_primal_residual)/ (norm(current_primal_residual)^2 + eps())

    primal_step_size_v .+= primal_step_size_gradient.^2
    dual_step_size_v .+= dual_step_size_gradient.^2

    primal_step_size .-= learning_rate * primal_step_size_gradient ./ sqrt.(primal_step_size_v .+ eps())
    dual_step_size .+= learning_rate * dual_step_size_gradient ./ sqrt.(dual_step_size_v .+ eps())

end

"""
`optimize(params::PrimalDualOptimizerParameters,
          problem::QuadraticProgrammingProblem)`
Solves a linear program using primal-dual hybrid gradient or
extragradient. If the step_size
specified in params is negative, picks a step size that ensures
step_size^2 * norm(constraint_matrix)^2 < 1,
a condition that guarantees provable convergence.
# Arguments
- `params::PrimalDualOptimizerParameters`: parameters.
- `original_problem::QuadraticProgrammingProblem`: the QP to solve.
# Returns
A PrimalDualOutput struct containing the solution found.
"""
function optimize(
  params::PrimalDualOptimizerParameters,
  problem::LinearProgrammingProblem,
)
    if params.rescale_flag
      scaled_problem = rescale_problem(
        10,
        true,
        4,
        problem,
      )
      scaled_lp = scaled_problem.scaled_lp
    else
      scaled_problem = rescale_problem(
        0,
        false,
        0,
        problem,
      )
      scaled_lp = scaled_problem.scaled_lp
    end

    primal_size = length(scaled_lp.variable_lower_bound)
    dual_size = length(scaled_lp.right_hand_side)
    
    if isnothing(params.step_size)
      step_size = 0.99 / opnorm(Matrix(scaled_lp.constraint_matrix))
    else
      step_size = params.step_size
    end

    if params.online_scaling
        primal_step_size = step_size * ones(primal_size)
        dual_step_size = step_size * ones(dual_size)
        # primal_step_size = zeros(primal_size)
        # dual_step_size = zeros(dual_size)
        primal_step_size_v = zeros(primal_size)
        dual_step_size_v = zeros(dual_size)
        learning_rate = params.learning_rate
    else
        primal_step_size = step_size
        dual_step_size = step_size
    end
  
    iteration_limit = params.iteration_limit
    stats = create_stats_data_frame()
  
    current_primal_solution = params.initial_primal_solution
    current_dual_solution = params.initial_dual_solution
    current_primal_residual = compute_primal_residual(current_primal_solution,problem)
    current_dual_residual = compute_dual_residual(current_dual_solution,problem)
    primal_delta, dual_delta = zeros(primal_size), zeros(dual_size)

    iteration = 0
    display_iteration_stats_heading()
    cumulative_kkt_passes = 0.0
    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0 # MV times per iteration

    start_time = time()
    while true
      # store stats and log
      terminate = false
      if iteration >= iteration_limit
        if params.verbosity
          println("Iteration limit reached")
        end
        terminate = true
      end
      
      store_stats = mod(iteration, params.record_every) == 0 || terminate 

      print_stats = params.verbosity && (
          mod(iteration, params.record_every * params.print_every) == 0 ||
          terminate
        )
        
      if store_stats
        this_iteration_stats = evaluate_unscaled_iteration_stats(
          scaled_problem,
          iteration,
          cumulative_kkt_passes,
          time() - start_time,
          current_primal_solution,
          current_dual_solution,
          primal_delta,
          dual_delta,
        )
        kkt_error = this_iteration_stats.kkt_error[end]
        
        if kkt_error <= params.kkt_tolerance
          if params.verbosity
            println("Found optimal solution after $iteration iterations")
          end
          terminate = true
        elseif kkt_error > 1e10
          if params.verbosity
            println("KKT error exploded after $iteration iterations")
          end
          terminate = true
        end
      end
      if store_stats
        append!(stats, this_iteration_stats)
      end
      if print_stats
        display_iteration_stats(this_iteration_stats)
        #log_iteration(problem, this_iteration_stats)
      end
  
      if terminate
        original_primal_solution::Vector{Float64} =
            current_primal_solution ./ scaled_problem.variable_rescaling
        original_dual_solution::Vector{Float64} =
            current_dual_solution ./ scaled_problem.constraint_rescaling
        
        optimize_output = PrimalDualOutput(
          original_primal_solution, 
          original_dual_solution,
          stats,
          time() - start_time,
          iteration,
          primal_size,
          dual_size,
          iteration >= iteration_limit ? STATUS_ITERATION_LIMIT : STATUS_OPTIMAL,
        )
        return optimize_output
      end
      iteration += 1
  
      
      next_primal, next_dual, next_primal_residual, next_dual_residual = take_pdhg_step(
        scaled_lp,
        current_primal_solution,
        current_dual_solution,
        current_primal_residual,
        current_dual_residual,
        primal_step_size,
        dual_step_size,
      )
      cumulative_kkt_passes +=  KKT_PASSES_PER_TERMINATION_EVALUATION

      if params.online_scaling
        step_size_update!(
            primal_step_size,
            dual_step_size,
            learning_rate*step_size, # learning_rate, 0.1/norm(A)
            primal_step_size_v,
            dual_step_size_v,
            current_primal_residual,
            current_dual_residual,
            next_primal_residual,
            next_dual_residual,
        )
      end

      primal_delta = next_primal .- current_primal_solution
      dual_delta = next_dual .- current_dual_solution
      # update iterates
      current_primal_solution = next_primal
      current_dual_solution = next_dual
      current_primal_residual = next_primal_residual
      current_dual_residual = next_dual_residual
    end
end


function solve(
  problem::LinearProgrammingProblem,
  iteration_limit::Int64,
  kkt_tolerance::Float64,
  initial_primal_solution::Vector{Float64},
  initial_dual_solution::Vector{Float64},
  online_scaling::Bool,
  learning_rate::Float64,
)

  #status = MOI.OTHER_ERROR
  println("solving problem with: ")
  print("rows = ", size(problem.constraint_matrix, 1), ", ")
  print("cols = ", size(problem.constraint_matrix, 2), ", ")
  println("nnz = ", length(problem.constraint_matrix.nzval), ".")
  println()

  params = PrimalDualOptimizerParameters(
    nothing, # step_size (forces the solver to use a provably correct step size)
    true, # rescaling
    5, # record every
    4, # print every
    true, # verbose
    iteration_limit, # iteration limit
    kkt_tolerance, # kkt tolerance
    initial_primal_solution, # initial primal solution
    initial_dual_solution,  # initial dual solution
    online_scaling, # bool: whether to use online saling
    learning_rate, # learning rate for online scaling
  )

  solver_output = optimize(params, problem)
  
  if params.online_scaling
    println("Using solver_plus.jl: Online Scaling TRUE!")
  else
    println("Using solver_plus.jl: Online Scaling FALSE!")
  end

  return solver_output
end




