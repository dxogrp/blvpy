# BLVPy: Disciplined Bilevel Programming

BLVPy is a [CVXPY](https://www.cvxpy.org/) extension for modeling and locally
solving optimistic bilevel programs. The lower-level problem is written as a
disciplined parametrized program (DPP), with selected upper variables declared
as its parameters. BLVPy canonicalizes that family once and replaces
lower-level optimality with primal feasibility, dual feasibility, and a conic
duality-gap constraint.

For a canonical lower problem

$$
\mathop{\mathrm{minimize}}_u\;c(x)^T u+d(x)
\quad\text{subject to}\quad A(x)u+s=b(x),\;s\in K,
$$

the relaxed single-level problem enforces

$$
s^T\lambda\leq\epsilon
$$

together with primal and dual conic feasibility. A warm-started continuation
method then decreases $\epsilon$. When those feasibility conditions hold, the
gap bounds lower-level objective suboptimality by $\epsilon$.

> [!IMPORTANT]
> BLVPy uses a local nonlinear solver. A successful solve is not a certificate
> of global upper-level optimality. Inspect the reported feasibility and gap
> residuals; solver termination alone is not a bilevel certificate.

## Supported models

The initial release supports exact lower-problem canonicalizations containing
only:

- zero cones (equalities),
- nonnegative cones (linear inequalities), and
- second-order cones.

This includes LPs, losslessly conic-representable QPs, and SOCPs. Models that
canonicalize to positive-semidefinite, exponential, or power cones are rejected
before the nonlinear solve. Lower problems must be real, continuous,
minimization problems and DPP with respect to their linked parameters. BLVPy
implements optimistic semantics: if the lower problem has several minimizers,
the upper problem may select the one most favorable to it.

Ordinary CVXPY parameters may also occur as fixed data. Their values are
snapshotted when the lower problem is first canonicalized; construct a new
`BilevelProblem` if those values need to change.

For the paper's pointwise graph contract, the initial exact-atom audit accepts
affine expressions plus absolute value, Huber loss, scalar/vector maxima and
minima, 1/2/infinity norms, quadratic forms and squares,
quadratic-over-linear, and sum-largest graphs. A model may canonicalize to an
SOCP yet still be rejected when its source atom has not been audited; this
conservative policy avoids treating cone type alone as a losslessness proof.

## Installation

BLVPy requires Python 3.12 or newer, CVXPY 1.9 or newer, and IPOPT. Install
the native IPOPT library for your operating system first, then install BLVPy:

```shell
pip install blvpy
```

The required `cyipopt` Python binding is installed with BLVPy. It needs the
native IPOPT library to be present while BLVPy is installed and used.

Canonicalization and lower-level initialization use Clarabel, which is included
with CVXPY.

## Basic usage

The same CVXPY variable appears directly in both levels. Declare which upper
variables parameterize the lower problem with `LowerProblem`:

```python
import cvxpy as cp

from blvpy import BilevelProblem, LowerProblem

x = cp.Variable(name="x")
y = cp.Variable(name="y")

lower = LowerProblem(
    cp.Minimize(cp.square(y - x)),
    parameters=[x],
)
bilevel = BilevelProblem(
    outer_objective=cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
    lower_problem=lower,
)

assert bilevel.is_dbp()
result = bilevel.solve(
    epsilon_initial=1e-1,
    epsilon_target=1e-6,
)

print(result.status)
print(x.value, y.value)
print(result.residuals)
```

Every variable listed in `LowerProblem(parameters=...)` is an upper-level
variable and is held fixed while the lower problem is solved. BLVPy uses one
deterministic upper start by default: it preserves an existing `.value`, uses
the midpoint of finite bounds, or otherwise uses zero clipped to any one-sided
bounds. It then attempts to project that point onto DCP upper constraints.

Set `starts` above one to enable randomized multistart. Only components with
finite two-sided CVXPY bounds are randomized; other components retain their
deterministic values. If automatic initialization and feasibility restoration
both fail, BLVPy names the variables whose `.value` should be initialized.

Passing a raw `cp.Problem` with `parameter_map` remains available through
BLVPy 0.1.x with a `FutureWarning`; it will be removed in 0.2.0.

`validate()` raises a detailed modeling error, while `is_dbp()` provides the
corresponding boolean check. `canonicalize()` exposes immutable cone-program
metadata for diagnostics. `solve()` returns the final source and canonical
variables, solver and continuation histories, complementarity, and separate
upper, recovery, primal, dual, cone, and gap residuals.

`result.epsilon_history` contains only accepted, strictly decreasing
tolerances. `result.attempted_epsilon_history` also includes failed solves and
retry points. `result.certified` remains false because numerical NLP output is
local and residual-based, not a rigorous finite-precision certificate.

## Examples

The [`examples`](examples) directory contains:

- [`analytic_quadratic.py`](examples/analytic_quadratic.py), whose exact
  bilevel solution is known;
- [`optimistic_lp.py`](examples/optimistic_lp.py), illustrating selection from
  a set of lower-level minimizers;
- [`parameter_dependent_socp.py`](examples/parameter_dependent_socp.py), with
  an upper-dependent constraint matrix and a second-order cone;
- [`multistart_restoration.py`](examples/multistart_restoration.py), showing
  automatic bounded sampling and feasibility restoration; and
- [`continuation_reproducibility.py`](examples/continuation_reproducibility.py),
  exposing accepted tolerances, retries, and seeded reproducibility.

Run an example with:

```shell
uv run python examples/analytic_quadratic.py
```

## Development

BLVPy follows the `uv` and Hatchling workflow used by the other `dxogrp`
CVXPY extensions:

```shell
make sync       # create the environment and install development dependencies
make test       # run the full test suite
make lint       # run Ruff without modifying files
make build      # build the source and wheel distributions
```

CI runs tests, linting, and package builds on Linux with IPOPT for Python 3.12
through 3.14. macOS and Windows jobs build the distributions without resolving
runtime dependencies, so packaging remains portable without requiring IPOPT on
those runners.

## License

BLVPy is licensed under the [Apache License 2.0](LICENSE).
