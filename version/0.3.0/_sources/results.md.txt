# Results and diagnostics

## Bilevel results

{meth}`blvpy.BilevelProblem.solve` returns a
{class}`blvpy.BilevelResult`.
Use `result.succeeded` as the main success check. Important fields include:

- `status`, `objective`, and `message`;
- immutable `variable_values` snapshots keyed by the original CVXPY variables;
- canonical primal, slack, and dual snapshots;
- `residuals`, `complementarity`, and `final_epsilon` conveniences;
- accepted and attempted epsilon histories;
- all {class}`blvpy.RunRecord` objects and `selected_run` for a best-of search.

Every upper `objective` stored in an {class}`~blvpy.IterationRecord`,
{class}`~blvpy.RunRecord`, or {class}`~blvpy.BilevelResult` is evaluated in the
modeled sense. Thus a `cp.Maximize` value is not negated in the records,
`all_objectives`, or progress output. The original lower objective object is
likewise available unchanged as {attr}`blvpy.LowerProblem.objective`.

The following statuses can be produced by {meth}`blvpy.BilevelProblem.solve`.
The **scope** column distinguishes the top-level result from a best-of run and
an individual epsilon attempt.

```{list-table}
:header-rows: 1
:widths: 18 15 57 10

* - Status
  - Scope
  - Meaning
  - Successful?
* - `optimal`
  - result, run, attempt
  - The returned point passed the solver and BLVPY's independent residual
    checks.
  - Yes
* - `optimal_inaccurate`
  - attempt
  - CVXPY reported a numerically less accurate solution that still passed
    BLVPY's residual checks. BLVPY may accept the attempt, but it recomputes
    the selected run's final diagnostics before returning the result.
  - Yes
* - `continuation_failed`
  - result, run
  - The run initialized, but continuation did not reach the requested target
    epsilon. The top-level result contains the best partial run when no run
    reached the target.
  - No
* - `initialization_failed`
  - run
  - This run failed before BLVPY accepted its initial-epsilon point. Other
    best-of runs may still continue.
  - No
* - `residual_check_failed`
  - run, attempt
  - A solver returned a point, but BLVPY's independently recomputed
    feasibility or relaxed-gap residuals exceeded the tolerance.
  - No
* - `objective_unavailable`
  - run, final attempt
  - A target-epsilon point was available, but its upper objective was missing
    or nonfinite.
  - No
* - `solver_error`
  - run, attempt
  - The selected nonlinear backend raised an ordinary solve error or failed
    to provide a status.
  - No
* - `user_limit`
  - run, attempt
  - CVXPY reports that the solver stopped at a user limit, such as an
    iteration or time limit.
  - No
* - `infeasible`
  - run, attempt
  - CVXPY reports the lifted problem as infeasible.
  - No
* - `infeasible_inaccurate`
  - run, attempt
  - CVXPY reports likely infeasibility, but with reduced numerical confidence.
  - No
* - `unbounded`
  - run, attempt
  - CVXPY reports the lifted problem as unbounded.
  - No
* - `unbounded_inaccurate`
  - run, attempt
  - CVXPY reports likely unboundedness, but with reduced numerical confidence.
  - No
* - `infeasible_or_unbounded`
  - run, attempt
  - The solver could not distinguish infeasibility from unboundedness.
  - No
```

Use `result.succeeded` rather than matching status strings in application code.
For `continuation_failed`, inspect `result.message`, `result.selected_run`, and
the per-run histories; its top-level snapshots describe the best available
partial point, not a successful target-epsilon solution.

Not every failure produces a result. Invalid models or settings, an unavailable
backend, and failure of every run before an initial point is accepted raise a
BLVPY exception instead. In particular, `initialization_failed` records are
available only when another run initialized far enough for BLVPY to return a
{class}`blvpy.BilevelResult`.

## Residuals

At a returned upper point $x$, BLVPY writes the canonical lower problem as

$$
\begin{array}{ll}
\text{minimize} & c(x)^T u+d(x)\\
\text{subject to} & A(x)u+s=b(x)\\
  & s\in\mathcal{K},
\end{array}
$$

where $u$ is the canonical lower primal vector, $s$ is the conic slack
variable, and $\mathcal{K}$ is the product of cones.
For a lower `cp.Minimize(f)`, the canonical objective represents $f$. For a
lower `cp.Maximize(f)`, it represents $-f$.
The numerical arrays
$A(x)$, $b(x)$, $c(x)$, and $d(x)$ are evaluated at the returned upper point.
Let $\lambda$ be the equality dual vector
that should belong to the dual cone $\mathcal{K}^*$.

These objects are not additional inputs that the user must provide. CVXPY's
canonicalization produces $A$, $b$, $c$, $d$, and $\mathcal K$ from the
modeled lower objective and constraints, and BLVPY evaluates their dependence
on $x$. In the returned result, `canonical_primal` stores $u$, `slack` stores
$s$, and `dual` stores $\lambda$. BLVPY recomputes the residuals from these
snapshots after each nonlinear attempt; no additional solver call is required.

{class}`blvpy.Residuals` reports how closely the returned numerical point
satisfies this canonical system and the original bilevel model:

```{list-table}
:header-rows: 1
:widths: 22 28 50

* - Field
  - Mathematical quantity
  - Interpretation
* - `primal_equality`
  - $\|A(x)u+s-b(x)\|_2$
  - Violation of the canonical lower equality. It is zero when the primal
    vector and slack satisfy the conic equations.
* - `dual_equality`
  - $\|A(x)^T\lambda+c(x)\|_2$
  - Violation of lower-level stationarity with respect to $u$.
* - `recovery`
  - $\max_i\|y_i-(R_i u+r_i)\|_2$
  - Checks that every returned source lower variable $y_i$ agrees with the
    affine value recovered from the canonical vector. Canonicalization produces
    the matrices $R_i$ and offsets $r_i$.
* - `upper_constraints`
  - $\max_j\operatorname{violation}(g_j(x,y))$
  - Largest CVXPY violation norm among the upper constraints and generated
    linked-variable domain constraints.
* - `primal_cone`
  - $\operatorname{dist}(s,\mathcal{K})$
  - Distance of the slack from the primal product cone.
* - `dual_cone`
  - $\operatorname{dist}(\lambda,\mathcal{K}^*)$
  - Distance of the dual vector from the dual product cone.
* - `complementarity`
  - $s^T\lambda$
  - Signed primal-dual cone pairing. Exact lower optimality requires zero
    complementarity; a numerically infeasible point can make it slightly
    negative.
* - `gap_violation`
  - $\max(s^T\lambda-\epsilon,0)$
  - Amount by which complementarity exceeds the continuation relaxation
    $s^T\lambda\leq\epsilon$. It is zero whenever that relaxed inequality is
    satisfied.
```

All six feasibility residuals ideally equal zero. For a residual record $r$,
BLVPY defines the aggregate

$$
F=\max\{r_{\rm primal\_equality},r_{\rm dual\_equality},r_{\rm recovery},
r_{\rm upper\_constraints},r_{\rm primal\_cone},r_{\rm dual\_cone}\}
$$

and the relaxed-gap violation

$$
G=\max(s^T\lambda-\epsilon,0).
$$

The convenience property `max_feasibility` returns $F$, while `max_violation`
returns $\max(F,G)$. Thus `max_feasibility` answers “how far is this point from
the lifted feasibility and stationarity conditions?”, whereas `max_violation`
also asks whether it satisfies the current epsilon-gap relaxation.

The `feasibility_tolerance` passed to {meth}`blvpy.BilevelProblem.solve`
controls whether BLVPY accepts an attempt. During solving, the same tolerance
is used for $F$ and $G$. A user can later apply different thresholds without
resolving the problem:

```python
residuals = result.residuals
print(residuals.max_feasibility)  # F
print(residuals.gap_violation)  # G
print(residuals.max_violation)  # max(F, G)

acceptable = residuals.is_feasible(
    tolerance=1e-7,
    gap_tolerance=1e-6,
)
```

This check concerns the returned lifted lower-optimality conditions.

## Complete gap diagnostics

Call the convenience method only when the extra fixed-upper lower solve is
useful:

```python
diagnostics = problem.gap_diagnostics(result)
print(diagnostics.source_gap)
print(diagnostics.identity_error)
```

The method reconstructs canonical data at the result's upper point and checks
the inexact identity

$$
c^Tu+b^T\lambda
=s^T\lambda+u^Tr_d-\lambda^Tr_p,
$$

where $r_p=Au+s-b$ and $r_d=A^T\lambda+c$. It then performs one additional
fixed-upper conic solve and reports the sense-normalized **source gap**

$$
\operatorname{source\_gap}=
\begin{cases}
f_0(x,y_{\rm returned})-f_0^*(x),&\text{for }\texttt{cp.Minimize},\\
f_0^*(x)-f_0(x,y_{\rm returned}),&\text{for }\texttt{cp.Maximize}.
\end{cases}
$$

Both definitions measure suboptimality and are nonnegative in exact
arithmetic. Small negative values can occur from numerical tolerances and are
not clamped. The other objective terms in {class}`blvpy.GapDiagnostics` use
the normalized canonical minimization convention.

The diagnostic solve defaults to silent Clarabel, but it is configurable:

```python
diagnostics = problem.gap_diagnostics(
    result,
    solver=cp.SCS,
    solver_options={"eps": 1e-7},
    solver_verbose=False,
)
```

Diagnosis accepts successful results and complete `continuation_failed`
results. It snapshots and restores affected model state.
