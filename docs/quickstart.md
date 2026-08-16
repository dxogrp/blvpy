# Quick start

The target bilevel problem is

$$
\begin{array}{ll}
\text{minimize} & (x-1)^2+(y+1)^2 \\
\text{subject to} & x\geq -1, \\
  & y\in\mathop{\mathrm{argmin}}_z (z-x)^2
\end{array}
$$

with variables $x, y \in \mathbf{R}$.
For every fixed $x$, the lower problem has the unique response $y=x$.
Substitution gives the reduced upper objective $2x^2+2$, so the exact bilevel solution is $(x,y)=(0,0)$ with objective value $2$.
The CVXPY variable `y` below is shared by the lower model and the upper objective.

```python
import cvxpy as cp
import blvpy as bp

# x is selected by the upper problem; y is selected by the lower problem.
x = cp.Variable(name="x")
y = cp.Variable(name="y")

# Listing x in parameters makes it fixed data whenever the lower problem is
# checked or solved. The original y object remains shared with the upper model.
lower = bp.LowerProblem(
    cp.Minimize(cp.square(y - x)),
    parameters=[x],
)

problem = bp.BilevelProblem(
    cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
    lower,
    upper_constraints=[x >= -1.0],
)

# validate() raises a detailed exception if BLVPY cannot reformulate the model.
problem.validate()
assert problem.is_dbp()

# The default solve follows one deterministic epsilon-continuation path.
result = problem.solve()
if not result.succeeded:
    raise RuntimeError(result.message)

print(result.status)
print("x =", result.variable_values[x])
print("y =", result.variable_values[y])
print("maximum violation =", result.residuals.max_violation)

# This optional call performs one additional fixed-upper lower solve.
diagnostics = problem.gap_diagnostics(result)
print("signed source gap =", diagnostics.source_gap)
```
