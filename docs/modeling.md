# Problem modeling

## Optimistic semantics

Construct the lower problem with {class}`blvpy.LowerProblem`, then pass it to {class}`blvpy.BilevelProblem`.
The same lower decision-variable objects may appear in the upper objective and constraints.
BLVPY preserves those objects, so the upper problem can choose the most favorable member of a nonsingleton lower solution set.

```python
x = cp.Variable(name="x")
y = cp.Variable(name="y")

lower = bp.LowerProblem(
    cp.Minimize(0.0 * y),
    [y >= x, y <= 1.0],
    parameters=[x],
)
problem = bp.BilevelProblem(
    cp.Minimize(cp.square(x) + cp.square(y - 1.0)),
    lower,
    upper_constraints=[x >= 0.0, x <= 1.0],
)
```

Every variable in `LowerProblem.parameters` is an upper variable that BLVPY holds fixed inside the lower problem.
BLVPY clones the lower expression tree and replaces each listed variable with a generated CVXPY parameter of matching shape and domain.
Unlisted variables remain the original lower variables.

## Structural requirements

The complete model must satisfy **all** of the following:

- The upper problem is a real-valued continuous minimization problem.
- The assembled upper objective and upper constraints are compatible with CVXPY's DNLP rules.
- The lower problem is a real-valued continuous convex minimization problem.
- The lower problem contains at least one canonical optimization variable; constant-only lower problems are not supported.
- The lower problem is DCP and DPP with respect to every linked upper variable.
- Every unlinked CVXPY parameter already has a finite value.
- CVXPY produces only zero, nonnegative, and second-order cone blocks when canonicalization is requested with a linear conic objective.

This includes linear programs, quadratic programs that CVXPY converts exactly to the accepted conic form, and second-order cone programs.

Call {meth}`blvpy.BilevelProblem.validate` to obtain a specific exception for an unsupported model.
See {doc}`troubleshooting` for the exception categories.

## Parameters and fixed data

The word `parameters` in {class}`~blvpy.LowerProblem` refers to upper CVXPY *variables*.
Ordinary CVXPY parameters may still appear as fixed lower data, but they must have values when the model is first canonicalized.
BLVPY freezes those values into the canonical family.
Change them only by constructing and canonicalizing a new {class}`~blvpy.BilevelProblem`.

Linked variables may be scalar, vector, or matrix valued.
Their relevant CVXPY domain attributes and native `bounds=` data are copied to the generated parameters and enforced by the lifted model.

## Variable values and bounds

Variable names are optional; CVXPY creates names automatically.
Explicit names are recommended because they make initialization and validation messages more
useful.

A pre-solve `.value` is only an initialization hint.
In ordinary deterministic solving, BLVPY uses an existing value, otherwise a point derived from finite bounds, otherwise zero, followed by variable-attribute projection and a best-effort projection onto DCP upper constraints.

The dynamically assigned `variable.sample_bounds` attribute is sampling-only metadata used by explicit `best_of` searches.
See {ref}`best-of-search`.

## Numerical backends

[IPOPT](https://coin-or.github.io/Ipopt/) is the default DNLP backend.
A different backend accepted by CVXPY's `nlp=True` solve path can be passed to {meth}`blvpy.BilevelProblem.solve` after proper installation, but they are not fully tested.

[Clarabel](https://clarabel.org/) is the default conic backend for fixed-upper
lower solves, initialization, projection, and
{meth}`~blvpy.BilevelProblem.gap_diagnostics`.
Both the DNLP and conic solvers can be overridden per solve.
