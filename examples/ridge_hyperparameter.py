import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Choosing a Ridge Penalty from Validation Data

    Regularization is often selected by trying a grid of values. Here we pose
    the same task as a bilevel problem: a model developer (the **leader**)
    chooses the ridge penalty, and a training procedure (the **follower**)
    fits regression coefficients for that penalty. The developer anticipates
    the trained model and minimizes its validation error.
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

    return BilevelProblem, LowerProblem, Path, cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bilevel formulation

    Let $(X_{\mathrm{tr}},q_{\mathrm{tr}})$ and
    $(X_{\mathrm{val}},q_{\mathrm{val}})$ denote the training and validation
    samples. For a regularization weight
    $\alpha\in[10^{-4},10]$, the follower computes

    \[
    \beta(\alpha)\in\mathop{\mathrm{argmin}}_{\beta}
      (1/n_{\mathrm{tr}})
      \lVert X_{\mathrm{tr}}\beta-q_{\mathrm{tr}}\rVert_2^2
      +\alpha\lVert\beta\rVert_2^2.
    \]

    The leader then solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{\alpha,\beta} &
      (1/n_{\mathrm{val}})
      \lVert X_{\mathrm{val}}\beta-q_{\mathrm{val}}\rVert_2^2\\
      \text{subject to} &
      \beta\in\beta(\alpha).
    \end{array}
    \]

    The lower problem is a strongly convex quadratic program, so its response
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
    n_train = 24
    n_validation = 80

    true_coefficients = np.array([1.6, -1.2, 0.8, 0.0, -0.5, 0.35, 0.0, 0.6])
    training_features = rng.normal(size=(n_train, n_features))
    validation_features = rng.normal(size=(n_validation, n_features))
    feature_scale = np.std(training_features, axis=0)
    training_features = training_features / feature_scale
    validation_features = validation_features / feature_scale
    training_targets = training_features @ true_coefficients + rng.normal(scale=1.1, size=n_train)
    validation_targets = validation_features @ true_coefficients + rng.normal(scale=0.25, size=n_validation)
    return (
        n_features,
        n_train,
        n_validation,
        training_features,
        training_targets,
        validation_features,
        validation_targets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the bilevel model

    Listing `ridge_weight` in `LowerProblem(parameters=[...])` makes it fixed
    data for training while leaving it as the leader's decision. Note that its
    native CVXPY bounds are part of the mathematical model.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    cp,
    n_features,
    n_train,
    n_validation,
    training_features,
    training_targets,
    validation_features,
    validation_targets,
):
    ridge_weight = cp.Variable(
        nonneg=True,
        bounds=[1e-4, 10.0],
        name="ridge_weight",
    )
    coefficients = cp.Variable(n_features, name="regression_coefficients")

    training_loss = cp.sum_squares(training_features @ coefficients - training_targets) / n_train
    lower_problem = LowerProblem(
        cp.Minimize(training_loss + ridge_weight * cp.sum_squares(coefficients)),
        parameters=[ridge_weight],
    )
    validation_loss = cp.sum_squares(validation_features @ coefficients - validation_targets) / n_validation
    problem = BilevelProblem(cp.Minimize(validation_loss), lower_problem)
    return coefficients, problem, ridge_weight


@app.cell
def _(problem):
    epsilon_target = 1e-9
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=epsilon_target,
        verbose=True,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, epsilon_target, result


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
def _(
    cp,
    n_features,
    n_train,
    np,
    training_features,
    training_targets,
    validation_features,
    validation_targets,
):
    grid_weights = np.unique(np.concatenate((np.geomspace(1e-4, 10.0, 70), np.array([1.0]))))
    grid_parameter = cp.Parameter(nonneg=True, name="grid_ridge_weight")
    grid_coefficients = cp.Variable(n_features, name="grid_coefficients")
    grid_training_expression = cp.sum_squares(
        training_features @ grid_coefficients - training_targets
    ) / n_train + grid_parameter * cp.sum_squares(grid_coefficients)
    grid_problem = cp.Problem(cp.Minimize(grid_training_expression))

    grid_training_errors = []
    grid_validation_errors = []
    for _weight in grid_weights:
        grid_parameter.value = float(_weight)
        grid_problem.solve(solver=cp.CLARABEL, warm_start=True)
        _coefficient_value = np.asarray(grid_coefficients.value, dtype=float)
        grid_training_errors.append(
            float(np.mean(np.square(training_features @ _coefficient_value - training_targets)))
        )
        grid_validation_errors.append(
            float(np.mean(np.square(validation_features @ _coefficient_value - validation_targets)))
        )

    grid_training_errors = np.asarray(grid_training_errors)
    grid_validation_errors = np.asarray(grid_validation_errors)
    baseline_index = int(np.flatnonzero(np.isclose(grid_weights, 1.0))[0])
    baseline_validation_error = float(grid_validation_errors[baseline_index])
    return (
        baseline_validation_error,
        grid_training_errors,
        grid_validation_errors,
        grid_weights,
    )


@app.cell(hide_code=True)
def _(
    baseline_validation_error,
    diagnostics,
    mo,
    result,
    ridge_weight,
):
    assert result.succeeded, result.message
    assert float(result.objective) < baseline_validation_error - 0.1, (
        "The selected ridge weight did not improve validation error."
    )

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | selected ridge weight | {float(ridge_weight.value):.5f} |
    | validation MSE at $\alpha=1$ | {baseline_validation_error:.6f} |
    | BLVPY validation MSE | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The selected penalty substantially improves validation error over the
    conventional $\alpha=1$ baseline. Its nonzero value also illustrates the
    bias-variance tradeoff: the training curve worsens as regularization grows,
    while the validation curve initially improves. This remains a local
    numerical bilevel solution, not a global certificate.
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
    grid_training_errors,
    grid_validation_errors,
    grid_weights,
    plt,
    result,
    ridge_weight,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.semilogx(grid_weights, grid_training_errors, label="training MSE", color="#377eb8")
    axis.semilogx(grid_weights, grid_validation_errors, label="validation MSE", color="#e41a1c")
    axis.scatter(
        [float(ridge_weight.value)],
        [float(result.objective)],
        color="black",
        marker="*",
        s=130,
        zorder=5,
        label="BLVPY selection",
    )
    axis.set(
        xlabel=r"ridge weight $\alpha$",
        ylabel="mean squared error",
        title="Training and validation error along the ridge path",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    figure_path = figure_dir / "ridge_hyperparameter.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
