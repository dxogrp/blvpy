# Polishing

An epsilon-continuation solve returns a point that approximately satisfies the
lower optimality conditions. Polishing holds the result's upper variables
fixed and solves the canonical lower problem once more, without the
continuation relaxation. This produces a fresh lower response and a compact
summary that helps you decide whether to adopt or discard the candidate.

Polishing is non-mutating: it restores every affected CVXPY variable,
parameter, and lifted value before returning or raising, and it does not
change the supplied {class}`~blvpy.BilevelResult`.

## Basic usage

Call {meth}`blvpy.BilevelProblem.polish` with a result from the same problem:

```python
import cvxpy as cp

polished = problem.polish(
    result,
    solver=cp.CLARABEL,
    solver_options=None,
    verbose=True,
    solver_verbose=False,
)

print(polished.feasible)
print(polished.residuals.max_violation)
print(polished.feasibility_tolerance)
print(polished.objective)
print(polished.objective_improvement_ratio)
```

Successful results and complete `continuation_failed` results can be
polished. The source-variable snapshots in the result, rather than the
variables' current `.value` attributes, define the point to polish.

With the default `verbose=True`, the call immediately writes a summary like
this to standard error:

```text
-------------------------------------------------------------------------------
                                   Polishing
-------------------------------------------------------------------------------
(BLVPY) Result: feasible=true | objective=8.000e+00
(BLVPY)   improvement_ratio=2.000e-01
(BLVPY) Residuals: max_violation=5.000e-08 | feasibility_tolerance=1.000e-07
```

When the candidate is infeasible, the summary instead prints every residual
used by the feasibility decision and marks the largest checked value:

```text
(BLVPY) Result: feasible=false | objective=8.000e+00
(BLVPY)   improvement_ratio=2.000e-01
(BLVPY) Residuals: feasibility_tolerance=1.000e-07
(BLVPY)   primal_equality=1.000e-08
(BLVPY)   dual_equality=2.000e-08
(BLVPY)   recovery=3.000e-08
(BLVPY)   upper_constraints=4.000e-04 | largest=true
(BLVPY)   primal_cone=5.000e-08
(BLVPY)   dual_cone=6.000e-08
(BLVPY)   gap_violation=7.000e-08
```

`largest=true` marks every exact tie; `max_violation` is omitted because it
would repeat the marked value.

Set `verbose=False` to suppress BLVPY's summary. This is independent of
`solver_verbose`: the latter controls CVXPY and native conic-solver output on
a best-effort basis.

## Interpreting the result

{class}`blvpy.PolishResult` is an immutable snapshot of the complete candidate,
its diagnostics, and its upper-objective comparison; see {doc}`api` for
individual field contracts. Its `feasible` property is derived rather than
stored:

```python
polished.feasible == polished.residuals.is_feasible(polished.feasibility_tolerance)
```

Both the original and polished upper objectives are reevaluated under the
same fixed-parameter state. For original objective $F_{\rm original}$ and
polished objective $F_{\rm polished}$, the reported ratio is

$$
\frac{F_{\rm original}-F_{\rm polished}}
     {|F_{\rm original}|}
\quad\text{for minimization},
$$

and

$$
\frac{F_{\rm polished}-F_{\rm original}}
     {|F_{\rm original}|}
\quad\text{for maximization}.
$$

A positive ratio is an improvement, zero is unchanged, and a negative ratio
is worse. When the original objective is exactly zero, division would be
undefined, so `objective_improvement_ratio` is `None` and the terminal output
shows `n/a`. A nonzero value that is merely close to zero is still used as the
denominator; the resulting ratio can therefore be large and should be read
alongside the absolute `objective`. If its magnitude exceeds floating-point
range, the ratio is `+inf` or `-inf`, with the usual better-or-worse sign.

## Feasibility

Polishing independently recomputes BLVPY's residuals at `epsilon=0` and checks
them against the originating solve's tolerance, because a successful solver
status indicates that numerical stopping rules were met rather than proving
exact feasibility. See {doc}`results` for the residual definitions and the
scope of polished and source-point diagnostics. For exponential- or power-cone
lower models, this recomputation can invoke the internal projection solvers
described there.

Raw `complementarity` remains available in `polished.residuals` but is omitted
from the terminal: at `epsilon=0`, its positive part is `gap_violation`, while
a negative value does not violate the one-sided gap check.

:::{warning}
If the lower problem has multiple optima, the conic solver chooses one of
them. That response can lose the optimistic tie-breaking achieved by the
bilevel solve, worsen the upper objective, or violate an upper constraint
that depends on lower variables. A completed polishing solve can therefore
return `feasible=False`.
:::

## Adopting the candidate

To adopt the complete candidate, project each immutable snapshot through the
original variable's attributes:

```python
if polished.feasible:
    for variable, value in polished.variable_values.items():
        variable.project_and_assign(value)
```

## Solver options and errors

Clarabel is the default fixed-upper conic solver. `solver_options` is copied
and forwarded to CVXPY, so backend-specific settings can be supplied without
being modified:

```python
polished = problem.polish(
    result,
    solver=cp.SCS,
    solver_options={"eps": 1e-7},
    solver_verbose=False,
)
```

Polishing raises rather than encoding solver-status details in
{class}`~blvpy.PolishResult`. An unavailable backend raises
{class}`blvpy.SolverUnavailableError`; a lower solve that fails or returns an
incomplete certificate raises {class}`blvpy.SolveError`. Invalid, incomplete,
or foreign results raise the corresponding argument error.
