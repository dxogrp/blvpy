# BLVPY: Disciplined Bilevel Programming

[![CI](https://github.com/dxogrp/blvpy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dxogrp/blvpy/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/blvpy.svg)](https://pypi.org/project/blvpy/) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dxogrp.github.io/blvpy/) [![License](https://img.shields.io/github/license/dxogrp/blvpy.svg)](https://github.com/dxogrp/blvpy/blob/main/LICENSE)

BLVPY is a [CVXPY](https://www.cvxpy.org/) extension for modeling and locally solving optimistic bilevel optimization problems.
A bilevel problem contains an optimization problem inside another optimization problem, i.e.,

$$
    \begin{array}{ll}
        \text{minimize} & F_0(x, y)\\
        \text{subject to} & F_i(x, y) \leq 0, \quad i = 1, \ldots, m\\
        & y \in S(x),
    \end{array}
$$

where $x \in \mathbf{R}^n$ contains the upper variables and $y \in \mathbf{R}^k$ contains lower variables constrained to belong to the set $S(x)$.
For a fixed $x \in \mathbf{R}^n$, the constraint set $S(x)$ is defined as the solution set of the following lower problem:

$$
    \begin{array}{rl}
      S(x) = \mathop{\rm argmin}_z & f_0(x, z)\\
      \text{subject to} & f_i(x, z) \leq 0, \quad i = 1, \ldots, p.
    \end{array}
$$

We say a bilevel problem is *disciplined bilevel programming* (DBLP) compatible if it satisfies the following conditions:

* The objective and constraint functions $F_i \colon \mathbf{R}^n \times \mathbf{R}^k \to \mathbf{R}$ for $i = 0, 1, \ldots, m$ of the upper problem are [DNLP](https://www.cvxpy.org/tutorial/dnlp/index.html)-compatible with variables $x \in \mathbf{R}^n$ and $y \in \mathbf{R}^k$.
* The objective and constraint functions $f_i \colon \mathbf{R}^n \times \mathbf{R}^k \to \mathbf{R}$ for $i = 0, 1, \ldots, p$ of the lower problem are [DPP](https://www.cvxpy.org/tutorial/dpp/index.html)-compatible with variable $z \in \mathbf{R}^k$ (or $y \in \mathbf{R}^k$), so that the lower problem is a disciplined convex program, parameterized by $x \in \mathbf{R}^n$.

BLVPY supports the modeling and solving of DBLP-compliant problems and uses *optimistic semantics*, i.e., when the lower problem has multiple minimizers, the upper problem may select the one most favorable to its objective.

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

### Development setup

BLVPY manages its development environment with [uv](https://docs.astral.sh/uv/).
Before setting up the repository, you should install uv and IPOPT.

1. Clone the repository:

   ```shell
   git clone https://github.com/dxogrp/blvpy.git
   cd blvpy
   ```

2. Create the virtual environment and install the locked development
   dependencies:

   ```shell
   make sync
   ```

Run `make test` and `make lint` before contributing changes.

## Quick start

The following example models the bilevel problem

$$
\begin{array}{ll}
\text{minimize} & (x-1)^2+(y+1)^2 \\
\text{subject to} & x\geq -1, \\
  & y\in\mathop{\mathrm{argmin}}_z (z-x)^2
\end{array}
$$

with variables $x, y \in \mathbf{R}$.

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
    upper_constraints=[x >= -1.0],
)

# Check that BLVPY can construct its supported single-level reformulation.
assert problem.is_dblp()
problem.validate()

# Solve the problem.
result = problem.solve()

# Optionally perform one additional lower solve for detailed gap diagnostics.
diagnostics = problem.gap_diagnostics(result)
```

Every variable in `LowerProblem.parameters` is an upper-level variable and is replaced internally by a CVXPY parameter in the lower problem.
Unlisted lower variables remain the original CVXPY objects, so the upper objective can use the returned lower solution directly.

## Examples

The [`examples`](examples) directory contains several [Marimo](https://marimo.io/) notebooks for demonstrating the use of BLVPY.
Run

```shell
make marimo
```

to install Marimo and open the notebooks in your browser.

## Documentation

The complete user guide and API reference are available at [this page](https://dxogrp.github.io/blvpy/).
To build and preview the Sphinx documentation locally, run:

```shell
make docs
```

## License

BLVPY is licensed under the [Apache License 2.0](LICENSE).
