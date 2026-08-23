import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Best-of search for local bilevel solutions

    BLVPY solves a nonlinear single-level reformulation locally. A single
    solve can therefore converge to different solutions from different
    initial points. This example compares a deliberately chosen deterministic
    start with BLVPY's randomized `best_of` search.
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

    figure_directory = Path(__file__).resolve().parent / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    plt.style.use(Path(__file__).resolve().parent / "zhlatex.mplstyle")
    return BilevelProblem, LowerProblem, cp, figure_directory, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem formulation

    We solve the optimistic bilevel problem

    \[
    \begin{array}{ll}
    \mathop{\rm minimize}_{x,y} & (x^2-1)^2+0.2x \\
    \mathop{\rm subject\ to} &
      y\in\mathop{\rm argmin}_{z}\ (z-x)^2.
    \end{array}
    \]

    Here $x$ is the upper variable and $y$ is the lower variable. For every
    fixed $x$, the lower problem has the unique solution

    \[
      y^\star(x)=x.
    \]

    Substituting this response into the upper problem gives the tilted
    double-well function

    \[
      \phi(x)=(x^2-1)^2+0.2x.
    \]

    Its stationary points satisfy

    \[
      \phi'(x)=4x(x^2-1)+0.2=0.
    \]

    Two of these points are local minima: a worse positive solution near
    $x=0.973994$ with objective $0.197434$, and a better negative solution
    near $x=-1.024120$ with objective $-0.202440$. This known landscape lets
    us identify which local solution each numerical run reaches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify the model

    `LowerProblem(parameters=[x])` declares that the lower problem treats the
    upper variable $x$ as fixed data. The lower variable $y$ is retained in
    the bilevel model, while the upper objective creates the nonconvex
    double-well landscape used to demonstrate local solving.
    """)
    return


@app.cell
def _(BilevelProblem, LowerProblem, cp):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    lower = LowerProblem(
        cp.Minimize(cp.square(y - x)),
        parameters=[x],
    )
    problem = BilevelProblem(
        cp.Minimize(cp.square(cp.square(x) - 1.0) + 0.2 * x + 0.0 * y),
        lower,
    )
    return problem, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deterministic solve and randomized best-of search

    The ordinary solve preserves an explicit `x.value`, so setting it to
    $1$ starts the local solver in the positive well. We then attach
    `sample_bounds=(-2, 2)` and request five randomized runs. Sampling bounds
    only describe where initial points may be drawn; they do not constrain
    the mathematical variable.

    Every viable best-of run performs its own complete epsilon continuation.
    BLVPY compares their final target-epsilon objectives and returns the best
    acceptable run. Consequently, cost grows approximately linearly with
    `best_of`.
    """)
    return


@app.cell
def _(np, problem, x):
    epsilon_initial = 1e-2
    epsilon_target = 1e-5

    x.value = 1.0
    deterministic_result = problem.solve(
        epsilon_initial=epsilon_initial,
        epsilon_target=epsilon_target,
        verbose=False,
    )
    deterministic_diagnostics = problem.gap_diagnostics(deterministic_result)
    deterministic_x = float(np.asarray(deterministic_result.variable_values[x]))

    x.sample_bounds = (-2.0, 2.0)
    best_result = problem.solve(
        epsilon_initial=epsilon_initial,
        epsilon_target=epsilon_target,
        best_of=5,
        seed=42,
        verbose=False,
    )
    best_diagnostics = problem.gap_diagnostics(best_result)
    best_x = float(np.asarray(best_result.variable_values[x]))

    initial_x = np.array([float(np.asarray(run.initial_values[x])) for run in best_result.runs])
    run_objectives = np.array([run.objective for run in best_result.runs], dtype=float)

    assert deterministic_result.succeeded, deterministic_result.message
    assert best_result.succeeded, best_result.message
    assert len(best_result.runs) == 5 and all(run.succeeded for run in best_result.runs), (
        "Expected five viable best-of runs."
    )
    assert np.any(initial_x < 0.0) and np.any(initial_x > 0.0), "Expected sampled starts in both local basins."
    assert abs(deterministic_x - 0.973994) <= 3e-3, "The deterministic solve missed the positive minimum."
    assert abs(best_x - (-1.024120)) <= 3e-3, "Best-of did not select the negative minimum."
    assert best_result.objective <= deterministic_result.objective - 0.3, (
        "Best-of did not materially improve on the deterministic local solution."
    )
    assert np.isclose(best_result.objective, np.min(run_objectives), atol=1e-8), (
        "The selected result is not the best terminal run objective."
    )
    return (
        best_diagnostics,
        best_result,
        best_x,
        deterministic_diagnostics,
        deterministic_result,
        deterministic_x,
        initial_x,
        run_objectives,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing local outcomes

    The left panel shows where the two top-level solves finish on the known
    reduced objective. The right panel relates every randomized initial point
    to its terminal objective. Runs ending in the same well have nearly equal
    objectives even though their initial points differ.
    """)
    return


@app.cell
def _(
    best_result,
    best_x,
    deterministic_result,
    deterministic_x,
    figure_directory,
    initial_x,
    np,
    plt,
    run_objectives,
):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    _grid = np.linspace(-1.6, 1.6, 500)
    _landscape = (_grid**2 - 1.0) ** 2 + 0.2 * _grid
    axes[0].plot(_grid, _landscape, color="0.25", linewidth=2)
    axes[0].scatter(
        deterministic_x,
        deterministic_result.objective,
        color="tab:orange",
        marker="s",
        s=75,
        label="Deterministic",
        zorder=3,
    )
    axes[0].scatter(
        best_x,
        best_result.objective,
        color="tab:red",
        marker="*",
        s=150,
        label="Best-of selected",
        zorder=3,
    )
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel(r"$\phi(x) = (x^2-1)^2+0.2x$")
    axes[0].legend(fontsize=12, frameon=False)

    for _run, _initial, _objective in zip(best_result.runs, initial_x, run_objectives):
        _selected = _run.index == best_result.selected_run_index
        axes[1].scatter(
            _initial,
            _objective,
            color="tab:red" if _selected else "tab:blue",
            marker="*" if _selected else "o",
            s=120 if _selected else 50,
            zorder=3,
        )
        axes[1].annotate(
            f"run {_run.index + 1}",
            (_initial, _objective),
            xytext=(2, 6),
            textcoords="offset points",
            fontsize=10,
        )
    axes[1].set_xlabel("Sampled initial $x$")
    axes[1].set_ylabel("Terminal upper objective")
    axes[1].set_ylim(-0.25, 0.25)
    axes[1].set_xlim(-1.8, 1.8)

    figure.tight_layout()
    figure_path = figure_directory / "best_of_local_optima.pdf"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    best_diagnostics,
    best_result,
    best_x,
    deterministic_diagnostics,
    deterministic_result,
    deterministic_x,
    mo,
):
    mo.md(f"""
    ## Result and interpretation

    **Deterministic solve**

    - Status: `{deterministic_result.status}`
    - Solution: $x={deterministic_x:.6f}$
    - Objective: ${deterministic_result.objective:.6f}$
    - Final epsilon: ${deterministic_result.final_epsilon:.3e}$
    - Maximum lifted violation: ${deterministic_result.residuals.max_violation:.3e}$
    - Complementarity: ${deterministic_result.complementarity:.3e}$
    - Lower source gap: ${deterministic_diagnostics.source_gap:.3e}$

    **Best-of solve**

    - Status: `{best_result.status}`
    - Selected run: {best_result.selected_run_index + 1} of {len(best_result.runs)}
    - Solution: $x={best_x:.6f}$
    - Objective: ${best_result.objective:.6f}$
    - Final epsilon: ${best_result.final_epsilon:.3e}$
    - Maximum lifted violation: ${best_result.residuals.max_violation:.3e}$
    - Complementarity: ${best_result.complementarity:.3e}$
    - Lower source gap: ${best_diagnostics.source_gap:.3e}$

    The deterministic start reaches the worse positive local minimum. The
    sampled runs explore both basins, allowing `best_of` to return the better
    negative solution.
    """)
    return


if __name__ == "__main__":
    app.run()
