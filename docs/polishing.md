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

All exact ties for the largest checked residual are marked. The derived
`max_violation` is omitted from this expanded form because it would duplicate
the value of a marked entry.

Set `verbose=False` to suppress BLVPY's summary. This is independent of
`solver_verbose`: the latter controls CVXPY and native conic-solver output on
a best-effort basis.

## Interpreting the result

{class}`blvpy.PolishResult` is immutable and exposes the following values:

- `residuals` contains the independently evaluated residuals for the polished
  candidate.
- `feasibility_tolerance` is the tolerance inherited from the solve that
  produced the original result.
- `feasible` is the read-only result of
  `residuals.is_feasible(feasibility_tolerance)`.
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
there is no positive continuation allowance for complementarity. The seven
values that determine `feasible` are lower primal and dual equality errors,
source-variable recovery error, the largest upper or generated
linked-variable constraint violation, primal- and dual-cone distances, and
`gap_violation`. They are compared with `feasibility_tolerance`, which comes
from the solve that produced the original result.

A successful lower-solver status means that the candidate met that solver's
numerical stopping rules; it does not prove exact feasibility. BLVPY therefore
recomputes these residuals independently from the returned candidate. This
also checks upper constraints involving the lower response selected by the
solver, which can differ when the lower problem has multiple optima.

The retained {class}`~blvpy.Residuals` also contains raw `complementarity`.
At `epsilon=0`, `gap_violation` is its positive part: positive
complementarity is reported unchanged, while a negative numerical value is
clamped to zero. Only `gap_violation` directly enters the feasibility check,
so the terminal's expanded infeasible summary omits raw `complementarity` to
avoid reporting the same positive discrepancy twice. The raw value remains
available as `polished.residuals.complementarity` for detailed inspection.

These residuals describe the polished candidate. By contrast, the original
{class}`~blvpy.BilevelResult`'s residuals and
{meth}`~blvpy.BilevelProblem.gap_diagnostics` describe the unpolished source
point. Polishing does not call `gap_diagnostics()` or perform another solve to
populate its result.

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
