## 1205 record

implemented the online scaling algo in `solver_plus.jl`

- [x] bug 1: when online_scaling=false, the result from `solver_plus.jl` is differnt from `solver.jl`

- [x] bug 2: the results from `solver_plus.jl` are always the same!

output for `solver_plus.jl`, but no online scaling:
```
Found optimal solution after 7660 iterations
7660    1.5e+04  5.8e-01 | 8.1e-07  4.6e-07   2.4e-07 |  1.3e+01  2.3e+00  3.3e-01 |
Using solver_plus.jl: Online Scaling FALSE!
```

FOR BUG 2!

output for `solver_plus.jl`, with online scaling, initialized as 0, the results never change.
output for `solver_plus.jl`, with online scaling, initialized as `step_size`, the results are the same as above.

-> so i believe that the primal/dual stepsize is never updated!
solution: i changed `+=` to `.+=` and this seems to be the problem!

output for lr 0.1*step_size
```
7000    1.4e+04  7.1e-01 | 5.9e-07  1.0e-06   9.0e-07 |  1.3e+01  2.3e+00  3.3e-01 |
Found optimal solution after 7010 iterations
Using solver_plus.jl: Online Scaling TRUE!
```

FOR BUG 1!

the output for `solver.jl` is
```
Found optimal solution after 7660 iterations
7660    1.5e+04  5.4e-01 | 8.3e-07  3.8e-07   1.6e-07 |  1.3e+01  2.3e+00  3.3e-01 |
```
which is almost the same as above. so i think this is just caused by slight numerical problem!

## 1208 record

TODO list
- [x] change the `plot_result.jl`, the same instance with different params should be logged into different files.
- [x] change the `run_problem.jl`, and add interface to adjust the lr for online algo.
- [ ] test more instances and tuning!
- [ ] further optimize the performance, customized to julia
