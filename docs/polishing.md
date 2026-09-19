# Polishing

An epsilon-continuation solve returns a point that approximately satisfies the
lower optimality conditions. Polishing holds the result's upper variables
fixed and solves the canonical lower problem once more, without the
continuation relaxation. This produces a fresh lower response and a compact
summary that helps you decide whether to adopt or discard the candidate.

Polishing is deliberately non-mutating. It restores every affected CVXPY
variable, parameter, and lifted value before returning or raising, and it
does not change the supplied {class}`~blvpy.BilevelResult`.

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
```

Set `verbose=False` to suppress BLVPY's summary. This is independent of
`solver_verbose`: the latter controls CVXPY and native conic-solver output on
a best-effort basis.

## Interpreting the result

{class}`blvpy.PolishResult` is immutable and contains four fields:

- `feasible` is whether the complete polished candidate passes BLVPY's
  standard feasibility check.
- `objective` is the upper objective at the polished point, in the original
  modeled sense. A `cp.Maximize` objective is not negated.
- `objective_improvement_ratio` compares the polished objective with the
  original result point. Positive values mean improvement for both
  minimization and maximization.
- `variable_values` contains immutable snapshots for every original CVXPY
  variable in the problem. Upper values come from the supplied result and
  lower values come from the fixed-upper solve.

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

Polishing evaluates the normal BLVPY residual system with `epsilon=0`, so
there is no positive continuation allowance for complementarity. The check
includes lower primal and dual equalities, primal- and dual-cone membership,
source-variable recovery, complementarity, and the upper and generated
linked-variable constraints. `feasible` uses the `feasibility_tolerance` from
the solve that produced the original result.

The boolean is intentionally the only feasibility information stored in a
{class}`~blvpy.PolishResult`. Use the original result's residuals or
{meth}`~blvpy.BilevelProblem.gap_diagnostics` when detailed numerical
diagnostics are needed.

:::{warning}
If the lower problem has multiple optima, the conic solver chooses one of
them. That response can lose the optimistic tie-breaking achieved by the
bilevel solve, worsen the upper objective, or violate an upper constraint
that depends on lower variables. A completed polishing solve can therefore
return `feasible=False`.
:::

## Adopting the candidate

Because polishing is non-mutating, inspect its summary before assigning any
values. To adopt the complete candidate explicitly, project each immutable
snapshot through the original variable's attributes:

```python
if polished.feasible:
    for variable, value in polished.variable_values.items():
        variable.project_and_assign(value)
```

This loop assigns upper and lower variables. Omitting the loop leaves the
model exactly as it was before `polish()`.

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
or foreign results raise the corresponding argument error. Model state is
restored on every error path.
