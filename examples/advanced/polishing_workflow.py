import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polish a coarse bilevel solution

    A bilevel solve at a deliberately coarse continuation tolerance can be a
    useful starting point, but its lower decision need not be the exact lower
    response. This example solves such a problem, re-solves the lower level at
    the returned upper point, inspects the complete polished candidate, and
    adopts it explicitly.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from blvpy import BilevelProblem, LowerProblem

    _style_path = Path(__file__).resolve().parents[1] / "_shared" / "zhlatex.mplstyle"
    plt.style.use(_style_path)
    return BilevelProblem, LowerProblem, cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A unique lower response and two upper wells

    Consider

    \[
    \begin{array}{ll}
    \mathop{\rm minimize}_{x,y}
      & (x-1)^2
        +10\left(\left(1-2y\right)^2-1\right)^2
        +0.2\left(1-2y\right)+1 \\
    \mathop{\rm subject\ to}
      & y\in\mathop{\rm argmin}_{z}\ (z-x)^2, \\
      & -2\leq x\leq2.
    \end{array}
    \]

    The lower response is unique: $y^\star(x)=x$. The upper objective is
    nonconvex in $y$, with wells near $y=0$ and $y=1$. We initialize $x$ at
    zero so the automatic lower solve also starts at $y=0$.
    """)
    return


@app.cell
def _(BilevelProblem, LowerProblem, cp):
    x = cp.Variable(name="x", bounds=[-2.0, 2.0])
    y = cp.Variable(name="y")
    x.value = 0.0

    lower = LowerProblem(
        cp.Minimize(cp.square(y - x)),
        parameters=[x],
    )
    _transformed_y = 1.0 - 2.0 * y
    problem = BilevelProblem(
        cp.Minimize(
            cp.square(x - 1.0) + 10.0 * cp.square(cp.square(_transformed_y) - 1.0) + 0.2 * _transformed_y + 1.0
        ),
        lower,
    )
    return problem, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve once at a coarse tolerance

    For this quadratic lower problem, the continuation relaxation permits
    $(y-x)^2\leq\epsilon$. Setting both continuation endpoints to
    $\epsilon=1$ performs one nonlinear solve and deliberately leaves room
    for the local solver to remain in the well near $y=0$ while moving $x$
    toward one.
    """)
    return


@app.cell
def _(cp, np, problem, x, y):
    epsilon = 1.0
    result = problem.solve(
        epsilon_initial=epsilon,
        epsilon_target=epsilon,
        solver=cp.IPOPT,
        verbose=False,
    )

    assert result.succeeded, result.message
    assert result.objective is not None
    solve_x = float(np.asarray(result.variable_values[x]))
    solve_y = float(np.asarray(result.variable_values[y]))
    solve_state = np.array([x.value, y.value], dtype=float)
    np.testing.assert_allclose([solve_x, solve_y], [1.0, 0.001255], atol=2e-4, rtol=0.0)
    np.testing.assert_allclose(result.objective, 1.199749, atol=5e-4, rtol=0.0)
    return epsilon, result, solve_state, solve_x, solve_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Re-solve the lower level at the returned upper point

    `polish()` fixes the upper variable at the result snapshot and solves the
    canonical lower problem without a positive continuation allowance. It
    reports a new candidate but does not assign that candidate to the model.
    Here Clarabel recovers the unique response $y=x$.
    """)
    return


@app.cell
def _(cp, np, problem, result, solve_state, solve_x, x, y):
    polished = problem.polish(
        result,
        solver=cp.CLARABEL,
        verbose=False,
    )

    polished_x = float(np.asarray(polished.variable_values[x]))
    polished_y = float(np.asarray(polished.variable_values[y]))
    polished_named_values = {
        variable.name(): float(np.asarray(value)) for variable, value in polished.variable_values.items()
    }

    assert polished.feasible
    assert polished.objective_improvement_ratio is not None
    assert polished.objective_improvement_ratio > 0.3
    np.testing.assert_allclose([polished_x, polished_y], [1.0, 1.0], atol=2e-4, rtol=0.0)
    np.testing.assert_allclose(polished_x, solve_x, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(polished.objective, 0.8, atol=5e-4, rtol=0.0)
    np.testing.assert_allclose(polished.objective_improvement_ratio, 1.0 / 3.0, atol=5e-4, rtol=0.0)
    np.testing.assert_array_equal(np.array([x.value, y.value], dtype=float), solve_state)
    return polished, polished_named_values, polished_x, polished_y


@app.cell(hide_code=True)
def _(
    epsilon,
    mo,
    polished,
    polished_named_values,
    polished_x,
    polished_y,
    result,
    solve_x,
    solve_y,
):
    _snapshots = ", ".join(f"{name}: {value:.6f}" for name, value in polished_named_values.items())
    mo.md(f"""
    ## Compare the candidates

    | Candidate | $x$ | $y$ | Upper objective | Interpretation |
    |---|---:|---:|---:|---|
    | $\\epsilon={epsilon:.0f}$ solve | {solve_x:.4f} | {solve_y:.4f} | {result.objective:.4f} | Relaxed result |
    | Polish | {polished_x:.4f} | {polished_y:.4f} | {polished.objective:.4f} | Exact response |

    The immutable `PolishResult` exposes four fields:

    - `feasible`: `{str(polished.feasible).lower()}`
    - `objective`: `{polished.objective:.6f}`
    - `objective_improvement_ratio`: `{polished.objective_improvement_ratio:.6f}`
    - `variable_values`: `{{{_snapshots}}}`

    The positive ratio means that this polished minimization objective is
    about {100.0 * polished.objective_improvement_ratio:.1f}% lower relative
    to the original result's objective. That improvement is possible here
    because the deliberately coarse nonconvex solve stops in the worse local
    well; polishing does not generally improve the upper objective. The live
    CVXPY variables still hold the solve point, confirming that polishing
    itself is non-mutating.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The local upper landscape

    At the returned value of $x$, the original lower decision lies in the
    left well. Polishing moves it to the unique lower response in the lower
    right well.
    """)
    return


@app.cell
def _(np, plt, polished_x, polished_y, solve_y):
    _y_grid = np.linspace(-0.2, 1.2, 500)
    _transformed_grid = 1.0 - 2.0 * _y_grid
    _upper_grid = (polished_x - 1.0) ** 2 + 10.0 * (_transformed_grid**2 - 1.0) ** 2 + 0.2 * _transformed_grid + 1.0

    _figure, _axis = plt.subplots(figsize=(7.0, 4.0))
    _axis.plot(_y_grid, _upper_grid, color="0.25", linewidth=2)
    _axis.scatter(
        solve_y,
        (polished_x - 1.0) ** 2 + 10.0 * ((1.0 - 2.0 * solve_y) ** 2 - 1.0) ** 2 + 0.2 * (1.0 - 2.0 * solve_y) + 1.0,
        color="tab:orange",
        marker="s",
        s=70,
        label="Coarse solve",
        zorder=3,
    )
    _axis.scatter(
        polished_y,
        (polished_x - 1.0) ** 2
        + 10.0 * ((1.0 - 2.0 * polished_y) ** 2 - 1.0) ** 2
        + 0.2 * (1.0 - 2.0 * polished_y)
        + 1.0,
        color="tab:green",
        marker="*",
        s=130,
        label="Polished response",
        zorder=3,
    )
    _axis.set_xlabel("Lower decision $y$")
    _axis.set_ylabel("Upper objective at fixed $x$")
    _axis.legend(frameon=False)
    _figure.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decide, then adopt explicitly

    Improvement is not guaranteed. A different local landscape can make the
    exact lower response worse for the upper objective, and a solver-selected
    response from a nonunique lower problem can violate upper constraints.
    The feasibility flag and objective comparison should therefore be checked
    before assigning the returned snapshots.

    This example accepts the candidate only because it is feasible and has a
    positive improvement ratio. Adoption is an explicit operation on every
    original CVXPY variable.
    """)
    return


@app.cell
def _(np, polished, x, y):
    accept_polished = (
        polished.feasible
        and polished.objective_improvement_ratio is not None
        and polished.objective_improvement_ratio > 0.0
    )
    if accept_polished:
        for _variable, _value in polished.variable_values.items():
            _variable.project_and_assign(_value)

    adopted_values = np.array([x.value, y.value], dtype=float)
    np.testing.assert_allclose(adopted_values, [1.0, 1.0], atol=2e-4, rtol=0.0)
    return accept_polished, adopted_values


@app.cell(hide_code=True)
def _(accept_polished, adopted_values, mo):
    mo.md(f"""
    ## Adopted model state

    - Candidate accepted: `{str(accept_polished).lower()}`
    - Current upper variable: $x={adopted_values[0]:.6f}$
    - Current lower variable: $y={adopted_values[1]:.6f}$

    The model now contains the polished snapshots because the assignment loop
    was run explicitly, not because `polish()` mutated it.
    """)
    return


if __name__ == "__main__":
    app.run()
