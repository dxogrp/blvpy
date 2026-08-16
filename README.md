# BLVPY: Disciplined Bilevel Programming

BLVPY is a [CVXPY](https://www.cvxpy.org/) extension for modeling and locally solving optimistic bilevel optimization problems.
A bilevel problem contains an optimization problem inside another optimization problem, i.e.,

$$
    \begin{array}{ll}
        \text{minimize} & F_0(x, y)\\
        \text{subject to} & F_i(x, y) \leq 0, \quad i = 1, \ldots, m\\
        & y \in S(x),
    \end{array}
$$
where $x \in \mathbf{R}^n$ and $y \in \mathbf{R}^k$ are the upper problem variables.
For a fixed $x \in \mathbf{R}^n$, the constraint set $S(x)$ is defined as the solution set of the following lower problem:

$$
    \begin{array}{rl}
      S(x) = \mathop{\rm argmin}_z & f_0(x, z)\\
      \text{subject to} & f_i(x, z) \leq 0, \quad i = 1, \ldots, p.
    \end{array}
$$

We say a bilevel problem is *disciplined bilevel programming* (DBP) compatible if it satisfies the following conditions:

* The objective and constraint functions $F_i \colon \mathbf{R}^n \times \mathbf{R}^k \to \mathbf{R}$ for $i = 0, 1, \ldots, m$ of the upper problem are [DNLP](https://www.cvxpy.org/tutorial/dnlp/index.html)-compatible with variables $x \in \mathbf{R}^n$ and $y \in \mathbf{R}^k$.
* The objective and constraint functions $f_i \colon \mathbf{R}^n \times \mathbf{R}^k \to \mathbf{R}$ for $i = 0, 1, \ldots, p$ of the lower problem are [DPP](https://www.cvxpy.org/tutorial/dpp/index.html)-compatible with variable $z \in \mathbf{R}^k$, so that the inner problem is a disciplined convex program, parameterized by $x \in \mathbf{R}^n$.

BLVPY supports the modeling and solving of DBP-compliant problems and uses *optimistic semantics*, i.e., when the lower problem has multiple minimizers, the upper problem may select the one most favorable to its objective.

## Basic idea

BLVPY treats the upper variable $x$ as a parameter of the convex lower problem.
When that lower problem satisfies DPP, BLVPY uses the CVXPY canonicalization backend to canonicalize the parameterized lower problem family into a cone program whose data depend affinely on $x$.

BLVPY expresses optimality of the canonicalized lower problem through primal feasibility, dual feasibility, and the relaxed conic gap condition $s^T \lambda \leq \epsilon$.
This produces a single-level problem, which BLVPY solves through CVXPY's nonlinear interface while warm-starting a sequence of problems with progressively smaller values of $\epsilon \to 0$.

## Installation

BLVPY requires:

* Python 3.12 or newer;
* CVXPY 1.9 or newer; and
* a native [IPOPT](https://coin-or.github.io/Ipopt/INSTALL.html) installation.

Install the native IPOPT library first following the [installation guide](https://coin-or.github.io/Ipopt/INSTALL.html).
Then install BLVPY from PyPI:

```shell
pip install blvpy
```

CVXPY also exposes DNLP paths for KNITRO, UNO, and COPT; these solvers may be selected in BLVPY after proper installation, but are not tested officially.
The required default (and recommended) nonlinear solver is IPOPT, which is free and open-source.
[Clarabel](https://clarabel.org/stable/) is the default backend conic solver.

## Quick start

```python
import cvxpy as cp
import blvpy as bp

# Use ordinary CVXPY variables in expressions at both levels.
# Here, x is controlled by the upper problem and y by the lower problem.
x = cp.Variable(name="x")
y = cp.Variable(name="y")

# Define the convex lower problem. Listing x in parameters means that x is
# held fixed whenever the lower problem is solved.
lower = bp.LowerProblem(
    cp.Minimize(cp.square(y - x)),
    parameters=[x],
)

# Define the upper objective and its constraints. The variable y is
# shared with the lower problem, giving the upper problem access to its response.
problem = bp.BilevelProblem(
    cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
    lower,
    outer_constraints=[x >= -1.0],
)

# Check that BLVPY can construct its supported single-level reformulation.
assert problem.is_dbp()
problem.validate()

# Solve the problem.
result = problem.solve()

# Optionally perform one additional lower solve for detailed gap diagnostics.
diagnostics = problem.gap_diagnostics(result)
```

Every variable in `LowerProblem.parameters` is an upper-level variable and is replaced internally by a CVXPY parameter in the lower problem.
Unlisted lower variables remain the original CVXPY objects, so the upper objective can use the returned lower solution directly.

## Solving and results

BLVPY currently supports linear, quadratic, and second-order cone lower problems.
By default, BLVPY builds one deterministic initial upper point and runs epsilon continuation from $10^{-1}$ to $10^{-6}$.
It preserves existing variable `.value` assignments, otherwise uses finite-bound midpoints, one unit inside a one-sided bound, or zero for an unbounded variable.

Use CVXPY's sampling-only bound argument `sample_bounds` and `best_of` to compare several local solutions without adding mathematical bounds:

```python
x.sample_bounds = (-2.0, 2.0)
result = problem.solve(best_of=5, seed=7)
```

Each viable best-of initialization performs a complete independent continuation.
BLVPY selects the acceptable target-epsilon run with the lowest upper objective.

`problem.solve()` returns a `BilevelResult` object containing:

* status, objective, and source-variable snapshots;
* canonical primal, slack, and dual values;
* upper, recovery, primal, dual, cone, complementarity, and gap residuals;
* accepted and attempted epsilon histories; and
* all `RunRecord` histories plus the selected run.

Use `result.succeeded` for the normal success check.
If continuation cannot reach the requested target after at least one viable initialization, BLVPY returns `status="continuation_failed"` with the best partial run and its diagnostics.

`problem.gap_diagnostics(result)` performs one additional fixed-upper conic solve.
It reports the signed source-level lower suboptimality and the canonical primal-dual gap identity.

The `problem.solve()` and `problem.gap_diagnostics()` argument `solver` selects the nonlinear backend.
The argument `conic_solver` of `problem.solve()` instead selects the numerical conic backend.
Solver-specific option mappings are passed through to CVXPY (and the conic solver).

## Examples

The [`examples`](examples) directory contains several [Marimo](https://marimo.io/) notebooks for demonstrating the use of BLVPY.
Run

```shell
make marimo
```

to install Marimo and open the notebooks in your browser.

## License

BLVPY is licensed under the [Apache License 2.0](LICENSE).
