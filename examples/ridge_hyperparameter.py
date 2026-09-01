import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Choosing a Ridge Penalty from Validation Data

    Regularization is often selected by trying a grid of values. Here we pose
    the same task as a bilevel problem: the **upper problem** chooses the ridge
    penalty from validation performance, while the **lower problem** fits the
    regression coefficients for that penalty using the training data.
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

    plt.style.use(Path(__file__).resolve().parent / "zhlatex.mplstyle")
    return BilevelProblem, LowerProblem, Path, cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bilevel formulation

    Let $(X_{\mathrm{tr}},y_{\mathrm{tr}})$ and
    $(X_{\mathrm{val}},y_{\mathrm{val}})$ denote the training and validation
    samples, with $m_{\mathrm{tr}}$ and $m_{\mathrm{val}}$ observations,
    respectively. The bilevel problem is

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{\lambda,w} &
      (1/m_{\mathrm{val}})
      \lVert X_{\mathrm{val}}w-y_{\mathrm{val}}\rVert_2^2\\
    \mathop{\mathrm{subject\ to}} &
      10^{-4}\leq\lambda\leq10,\\
      &w\in S(\lambda),
    \end{array}
    \]

    where

    \[
    S(\lambda)=\mathop{\mathrm{argmin}}_w
      (1/m_{\mathrm{tr}})
      \lVert X_{\mathrm{tr}}w-y_{\mathrm{tr}}\rVert_2^2
      +\lambda\lVert w\rVert_2^2.
    \]

    The lower problem is strongly convex because $\lambda>0$, so its response
    is unique. BLVPY converts it to its exact SOCP representation and applies
    $\epsilon$-gap continuation to the optimistic single-level reformulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deterministic regression data

    We generate 8 features, 24 noisy training observations, and 80 validation
    observations. The validation set is larger and less noisy, so it provides
    a useful signal for choosing a nonzero regularization weight.
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(1907)
    n_features = 8
    m_tr = 24
    m_val = 80

    w_true = np.array([1.6, -1.2, 0.8, 0.0, -0.5, 0.35, 0.0, 0.6])
    X_tr = rng.normal(size=(m_tr, n_features))
    X_val = rng.normal(size=(m_val, n_features))
    feature_scale = np.std(X_tr, axis=0)
    X_tr = X_tr / feature_scale
    X_val = X_val / feature_scale
    y_tr = X_tr @ w_true + rng.normal(scale=1.1, size=m_tr)
    y_val = X_val @ w_true + rng.normal(scale=0.25, size=m_val)
    return X_tr, X_val, m_tr, m_val, n_features, y_tr, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the bilevel model

    Listing `lbd` in `LowerProblem(parameters=[...])` makes it fixed
    data for training while leaving it as a decision of the upper problem.
    Note that the native bounds for `lbd`
    are part of the mathematical model.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    X_tr,
    X_val,
    cp,
    m_tr,
    m_val,
    n_features,
    y_tr,
    y_val,
):
    lbd = cp.Variable(
        nonneg=True,
        bounds=[1e-4, 10.0],
        name="lbd",
    )
    w = cp.Variable(n_features, name="w")

    training_loss = cp.sum_squares(X_tr @ w - y_tr) / m_tr
    lower_problem = LowerProblem(
        cp.Minimize(training_loss + lbd * cp.sum_squares(w)),
        parameters=[lbd],
    )
    validation_loss = cp.sum_squares(X_val @ w - y_val) / m_val
    problem = BilevelProblem(cp.Minimize(validation_loss), lower_problem)
    return lbd, problem


@app.cell
def _(problem):
    epsilon_target = 1e-5
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=epsilon_target,
        verbose=True,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Direct-CVXPY comparison grid

    For context, we independently solve the fixed-penalty training problem on
    a logarithmic grid with Clarabel. Note that this grid is only a visualization
    and a baseline, i.e., it is not visible to BLVPY when choosing the penalty.
    """)
    return


@app.cell
def _(X_tr, X_val, cp, m_tr, m_val, n_features, np, y_tr, y_val):
    lbd_grid = np.unique(np.concatenate((np.geomspace(1e-4, 10.0, 70), np.array([1.0]))))
    lbd_parameter = cp.Parameter(nonneg=True, name="lbd_grid")
    w_grid = cp.Variable(n_features, name="w_grid")
    grid_lower_objective = cp.sum_squares(X_tr @ w_grid - y_tr) / m_tr + lbd_parameter * cp.sum_squares(w_grid)
    grid_problem = cp.Problem(cp.Minimize(grid_lower_objective))

    grid_training_mse = []
    grid_validation_mse = []
    for _lbd in lbd_grid:
        lbd_parameter.value = float(_lbd)
        grid_problem.solve(solver=cp.CLARABEL, warm_start=True)
        _w_value = np.asarray(w_grid.value, dtype=float)
        grid_training_mse.append(float(np.sum(np.square(X_tr @ _w_value - y_tr)) / m_tr))
        grid_validation_mse.append(float(np.sum(np.square(X_val @ _w_value - y_val)) / m_val))

    grid_training_mse = np.asarray(grid_training_mse)
    grid_validation_mse = np.asarray(grid_validation_mse)
    baseline_index = int(np.flatnonzero(np.isclose(lbd_grid, 1.0))[0])
    baseline_validation_mse = float(grid_validation_mse[baseline_index])
    return (
        baseline_validation_mse,
        grid_training_mse,
        grid_validation_mse,
        lbd_grid,
    )


@app.cell(hide_code=True)
def _(baseline_validation_mse, diagnostics, lbd, mo, result):
    assert result.succeeded, result.message
    assert float(result.objective) < baseline_validation_mse - 0.1, (
        "The selected ridge weight did not improve validation error."
    )

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | selected ridge weight | {float(lbd.value):.5f} |
    | validation MSE at $\lambda=1$ | {baseline_validation_mse:.6f} |
    | BLVPY validation MSE | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The selected penalty substantially improves validation error over the
    conventional $\lambda=1$ baseline. Its nonzero value also illustrates the
    bias-variance tradeoff: the training curve worsens as regularization grows,
    while the validation curve initially improves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization path

    The curves below come from the independent fixed-penalty solves. The marker
    shows the continuous penalty selected by BLVPY.
    """)
    return


@app.cell
def _(
    Path,
    grid_training_mse,
    grid_validation_mse,
    lbd,
    lbd_grid,
    plt,
    result,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.semilogx(lbd_grid, grid_training_mse, label="Training", color="C0")
    axis.semilogx(lbd_grid, grid_validation_mse, label="Validation", color="C3")
    axis.scatter(
        [float(lbd.value)],
        [float(result.objective)],
        color="black",
        marker="X",
        s=80,
        zorder=5,
    )
    axis.set(
        xlabel=r"$\lambda$",
        ylabel=r"$(1/m){\|Xw-y\|}_2^2$",
        xlim=(1e-4, 10),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=13)
    fig.tight_layout()
    figure_path = figure_dir / "ridge_hyperparameter.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
