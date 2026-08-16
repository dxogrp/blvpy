import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimistic selection from a lower linear program

    This example shows what *optimistic* bilevel semantics means when the
    lower-level problem has many optimal points.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import marimo as mo
    import numpy as np

    from blvpy import BilevelProblem, LowerProblem

    return BilevelProblem, LowerProblem, cp, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem formulation

    Consider

    \[
    \begin{array}{ll}
    \mathop{\rm minimize}_{x,y} & x^2 + (y-1)^2 \\
    \mathop{\rm subject\ to} & 0 \leq x \leq 1, \\
      & y \in \mathop{\rm argmin}_{z}\ 0 \\
      & \qquad\mathop{\rm subject\ to}\quad x \leq z \leq 1.
    \end{array}
    \]

    Every $z\in[x,1]$ is lower-optimal. Under optimistic semantics the upper
    level may select, among those tied lower optima, the value it prefers.
    It therefore chooses $y=1$, while $x^2$ selects $x=0$. The exact upper
    objective is zero.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the model

    The same CVXPY variable `y` appears in the lower constraints and upper
    objective. BLVPY preserves that shared variable when it constructs the
    single-level reformulation.
    """)
    return


@app.cell
def _(BilevelProblem, LowerProblem, cp):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    lower = LowerProblem(
        cp.Minimize(0.0 * y),
        [y >= x, y <= 1.0],
        parameters=[x],
    )
    problem = BilevelProblem(
        cp.Minimize(cp.square(x) + cp.square(y - 1.0)),
        lower,
        upper_constraints=[x >= 0.0, x <= 1.0],
    )
    return problem, x, y


@app.cell
def _(np, problem, x, y):
    epsilon_target = 1e-9
    result = problem.solve(epsilon_target=epsilon_target)
    diagnostics = problem.gap_diagnostics(result)

    assert result.succeeded, result.message
    np.testing.assert_allclose(
        [float(np.asarray(x.value)), float(np.asarray(y.value)), result.objective],
        [0.0, 1.0, 0.0],
        atol=5e-3,
        rtol=0.0,
    )
    return diagnostics, result


@app.cell(hide_code=True)
def _(diagnostics, mo, result, x, y):
    mo.md(rf"""
    ## Result and interpretation

    - Status: `{result.status}`
    - Final epsilon: ${result.final_epsilon:.1e}$
    - Upper variable: $x={float(x.value):.6f}$
    - Optimistically selected lower solution: $y={float(y.value):.6f}$
    - Upper objective: ${result.objective:.6f}$
    - Maximum lifted violation: ${result.residuals.max_violation:.3e}$
    - Complementarity: ${result.complementarity:.3e}$
    - Independently evaluated lower source gap: ${diagnostics.source_gap:.3e}$

    The lower source gap is essentially zero for every feasible $y$ in
    this example. The value $y\approx1$ is selected because it is the
    lower optimizer preferred by the upper objective.
    """)
    return


if __name__ == "__main__":
    app.run()
