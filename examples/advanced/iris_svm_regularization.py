import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regularizing a Linear SVM on the Iris Data

    This example uses bilevel optimization to choose the penalty parameter,
    and therefore the regularization tradeoff, of a linear support-vector
    machine (SVM) from validation performance.
    """)
    return


@app.cell
def _():
    import csv
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from blvpy import BilevelProblem, LowerProblem

    plt.style.use(Path(__file__).resolve().parents[1] / "_shared" / "zhlatex.mplstyle")
    return BilevelProblem, LowerProblem, Path, cp, csv, mo, np, plt


@app.cell(hide_code=True)
def _(Path, csv, np):
    data_path = Path(__file__).resolve().parents[1] / "_shared" / "data" / "iris.csv"
    feature_names = (
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    )
    class_names = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")

    with data_path.open(encoding="utf-8", newline="") as _stream:
        _rows = list(csv.DictReader(_stream))

    features = np.asarray(
        [[float(_row[_name]) for _name in feature_names] for _row in _rows],
        dtype=float,
    )
    species = np.asarray([_row["species"] for _row in _rows])

    assert features.shape == (150, 4), "Expected all 150 Iris measurements."
    assert np.all(np.isfinite(features)), "Iris measurements must be finite."
    assert all(np.count_nonzero(species == _name) == 50 for _name in class_names), (
        "Expected 50 observations from each species."
    )
    return class_names, features, species


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data source

    The vendored data are the corrected 150-row `bezdekIris.data` file from
    the [UCI Iris dataset](https://doi.org/10.24432/C56C76). The notebook
    reads the local copy and performs no runtime download.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A deterministic validation split

    The model uses Versicolor and Virginica as a binary classification task.
    Within each class, every fifth observation is reserved for validation.
    This gives 80 training and 20 validation observations without randomness.
    Feature means and scales are computed from the training set only.
    """)
    return


@app.cell
def _(class_names, features, np, species):
    _binary_mask = np.isin(species, class_names[1:])
    _binary_features = features[_binary_mask]
    _binary_species = species[_binary_mask]
    _binary_targets = np.where(_binary_species == class_names[2], 1.0, -1.0)

    _validation_mask = np.zeros(_binary_species.size, dtype=bool)
    for _class_name in class_names[1:]:
        _class_indices = np.flatnonzero(_binary_species == _class_name)
        _validation_mask[_class_indices[::5]] = True
    _training_mask = ~_validation_mask

    _training_features_raw = _binary_features[_training_mask]
    _validation_features_raw = _binary_features[_validation_mask]
    training_mean = np.mean(_training_features_raw, axis=0)
    training_scale = np.std(_training_features_raw, axis=0)
    X_training = (_training_features_raw - training_mean) / training_scale
    X_validation = (_validation_features_raw - training_mean) / training_scale
    y_training = _binary_targets[_training_mask]
    y_validation = _binary_targets[_validation_mask]

    m_training, n_features = X_training.shape
    m_validation = X_validation.shape[0]
    assert (m_training, m_validation, n_features) == (80, 20, 4)
    assert np.all(training_scale > 0.0)
    np.testing.assert_allclose(np.mean(X_training, axis=0), 0.0, atol=1e-12)
    return (
        X_training,
        X_validation,
        m_training,
        m_validation,
        n_features,
        y_training,
        y_validation,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target bilevel problem

    Let $(x_i^{\mathrm{tr}},y_i^{\mathrm{tr}})$ and
    $(x_j^{\mathrm{val}},y_j^{\mathrm{val}})$ denote the standardized training
    and validation samples, where $y\in\{-1,+1\}$, and define
    $[u]_+=\max\{u,0\}$. Here $m_{\mathrm{tr}}=80$ and
    $m_{\mathrm{val}}=20$; Versicolor has label $-1$ and Virginica has label
    $+1$.

    For a fixed penalty value $C$, the follower trains a linear
    squared-hinge SVM. Its response set is

    \[
    \begin{array}{ll}
    S(C)=\mathop{\mathrm{argmin}}_{\widetilde w,\widetilde b,\widetilde\xi} &
      \tfrac12\lVert\widetilde w\rVert_2^2
      +\dfrac{C}{m_{\mathrm{tr}}}\lVert\widetilde\xi\rVert_2^2\\
    \mathop{\mathrm{subject\ to}} &
      \widetilde\xi_i\geq
      1-y_i^{\mathrm{tr}}\left((x_i^{\mathrm{tr}})^T\widetilde w+\widetilde b\right),
      \quad i=1,\ldots,m_{\mathrm{tr}},\\
      & \widetilde\xi\succeq0.
    \end{array}
    \]

    The leader chooses $C$ from validation performance while requiring the
    classifier to be a follower response:

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{C,w,b,\xi} &
      \dfrac{1}{m_{\mathrm{val}}}
      \displaystyle\sum_{j=1}^{m_{\mathrm{val}}}
      \left[
        1-y_j^{\mathrm{val}}\left((x_j^{\mathrm{val}})^Tw+b\right)
      \right]_+^2
      +10^{-6}C^2\\
    \mathop{\mathrm{subject\ to}} &
      10^{-3}\leq C\leq100,\\
      &(w,b,\xi)\in S(C).
    \end{array}
    \]

    Thus, the upper level may select the hyperparameter $C$, but it may not
    choose the classifier freely: $(w,b,\xi)$ must come from SVM training on
    the training set. The small $10^{-6}C^2$ term discourages unnecessarily
    large values of $C$.
    """)
    return


@app.cell
def _(X_training, cp, m_training, n_features, np, y_training):
    _fixed_C = cp.Parameter(nonneg=True, name="fixed_C")
    _fixed_weights = cp.Variable(n_features, name="fixed_weights")
    _fixed_intercept = cp.Variable(name="fixed_intercept")
    _fixed_slack = cp.Variable(m_training, nonneg=True, name="fixed_slack")
    _fixed_problem = cp.Problem(
        cp.Minimize(
            0.5 * cp.sum_squares(_fixed_weights) + (_fixed_C / m_training) * cp.sum_squares(_fixed_slack),
        ),
        [
            _fixed_slack >= 1.0 - cp.multiply(y_training, X_training @ _fixed_weights + _fixed_intercept),
        ],
    )

    def solve_fixed_svm(fixed_C):
        _fixed_C.value = float(fixed_C)
        _fixed_problem.solve(solver=cp.CLARABEL, warm_start=True)
        assert _fixed_problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        return (
            np.asarray(_fixed_weights.value, dtype=float).copy(),
            float(_fixed_intercept.value),
            np.asarray(_fixed_slack.value, dtype=float).copy(),
        )

    return (solve_fixed_svm,)


@app.cell
def _(X_validation, np, solve_fixed_svm, y_validation):
    initial_C = 10.0
    initial_weights, initial_intercept, initial_slack = solve_fixed_svm(initial_C)
    _initial_margins = y_validation * (X_validation @ initial_weights + initial_intercept)
    initial_validation_loss = float(np.mean(np.maximum(0.0, 1.0 - _initial_margins) ** 2))
    return (
        initial_C,
        initial_intercept,
        initial_slack,
        initial_validation_loss,
        initial_weights,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the model

    Listing `C` in `LowerProblem(parameters=[...])` makes it fixed while the
    SVM is trained, but leaves it as an upper-level decision. We initialize
    every source variable from an ordinary CVXPY solve at $C=10$, then use
    BLVPY's epsilon continuation from $10^{-2}$ to $10^{-5}$.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    X_training,
    X_validation,
    cp,
    initial_C,
    initial_intercept,
    initial_slack,
    initial_weights,
    m_training,
    m_validation,
    n_features,
    y_training,
    y_validation,
):
    C = cp.Variable(nonneg=True, bounds=[1e-3, 100.0], name="C")
    weights = cp.Variable(n_features, name="weights")
    intercept = cp.Variable(name="intercept")
    slack = cp.Variable(m_training, nonneg=True, name="slack")

    C.value = initial_C
    weights.value = initial_weights
    intercept.value = initial_intercept
    slack.value = initial_slack

    lower_problem = LowerProblem(
        cp.Minimize(
            0.5 * cp.sum_squares(weights) + (C / m_training) * cp.sum_squares(slack),
        ),
        [
            slack >= 1.0 - cp.multiply(y_training, X_training @ weights + intercept),
        ],
        parameters=[C],
    )
    validation_hinge = cp.pos(1.0 - cp.multiply(y_validation, X_validation @ weights + intercept))
    problem = BilevelProblem(
        cp.Minimize(cp.sum_squares(validation_hinge) / m_validation + 1e-6 * cp.square(C)),
        lower_problem,
    )
    return C, intercept, problem, weights


@app.cell
def _(C, X_validation, intercept, np, problem, weights, y_validation):
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-5,
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)

    selected_C = float(np.asarray(result.variable_values[C]))
    selected_weights = np.asarray(result.variable_values[weights], dtype=float)
    selected_intercept = float(np.asarray(result.variable_values[intercept]))
    _selected_scores = X_validation @ selected_weights + selected_intercept
    _selected_margins = y_validation * _selected_scores
    selected_validation_loss = float(np.mean(np.maximum(0.0, 1.0 - _selected_margins) ** 2))
    validation_accuracy = float(np.mean(np.where(_selected_scores >= 0.0, 1.0, -1.0) == y_validation))
    return (
        diagnostics,
        result,
        selected_C,
        selected_intercept,
        selected_validation_loss,
        selected_weights,
        validation_accuracy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent fixed-$C$ checks

    For comparison, we solve 64 ordinary SVMs on a logarithmic grid using
    CVXPY. We also solve once at BLVPY's continuous value of $C$. These
    problems check the lower response and put the selected value on a familiar
    regularization path. The grid reference minimizes the same upper objective,
    including the $10^{-6}C^2$ stabilizer, while the plot shows its
    validation-loss component.
    """)
    return


@app.cell
def _(
    X_validation,
    np,
    selected_C,
    selected_intercept,
    selected_weights,
    solve_fixed_svm,
    y_validation,
):
    C_grid = np.geomspace(1e-3, 100.0, 64)
    grid_validation_loss = np.empty_like(C_grid)
    grid_upper_objective = np.empty_like(C_grid)
    for _index, _C_value in enumerate(C_grid):
        _grid_weights, _grid_intercept, _ = solve_fixed_svm(float(_C_value))
        _grid_margins = y_validation * (X_validation @ _grid_weights + _grid_intercept)
        grid_validation_loss[_index] = np.mean(np.maximum(0.0, 1.0 - _grid_margins) ** 2)
        grid_upper_objective[_index] = grid_validation_loss[_index] + 1e-6 * _C_value**2

    _reference_weights, _reference_intercept, _ = solve_fixed_svm(selected_C)
    _reference_margins = y_validation * (X_validation @ _reference_weights + _reference_intercept)
    reference_validation_loss = float(np.mean(np.maximum(0.0, 1.0 - _reference_margins) ** 2))
    reference_weight_error = float(np.max(np.abs(selected_weights - _reference_weights)))
    reference_intercept_error = abs(selected_intercept - _reference_intercept)

    _grid_best_index = int(np.argmin(grid_upper_objective))
    grid_best_C = float(C_grid[_grid_best_index])
    return (
        C_grid,
        grid_best_C,
        grid_validation_loss,
        reference_intercept_error,
        reference_validation_loss,
        reference_weight_error,
    )


@app.cell(hide_code=True)
def _(
    diagnostics,
    grid_best_C,
    initial_C,
    initial_validation_loss,
    mo,
    reference_intercept_error,
    reference_validation_loss,
    reference_weight_error,
    result,
    selected_C,
    selected_validation_loss,
    validation_accuracy,
):
    assert result.succeeded, result.message
    assert reference_weight_error <= 5e-3
    assert reference_intercept_error <= 5e-3
    assert abs(selected_validation_loss - reference_validation_loss) <= 1e-3

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | selected $C$ | {selected_C:.5f} |
    | validation squared-hinge loss | {selected_validation_loss:.6f} |
    | validation accuracy | {validation_accuracy:.1%} |
    | validation loss at initial $C={initial_C:g}$ | {initial_validation_loss:.6f} |
    | best 64-point grid value of $C$ | {grid_best_C:.5f} |
    | fixed-$C$ validation loss | {reference_validation_loss:.6f} |
    | maximum fixed-$C$ weight difference | {reference_weight_error:.3e} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The continuous bilevel choice lies near the best independent grid point
    and attains {validation_accuracy:.0%} accuracy on the deterministic
    validation split.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization path

    The curve shows validation loss from the independent fixed-$C$ solves,
    together with BLVPY's continuous selection.
    """)
    return


@app.cell(hide_code=True)
def _(
    C_grid,
    Path,
    grid_validation_loss,
    plt,
    selected_C,
    selected_validation_loss,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(6.4, 4.3))
    axes.semilogx(C_grid, grid_validation_loss, color="C3", linewidth=2, label="Fixed-$C$ solves")
    axes.scatter(
        selected_C,
        selected_validation_loss,
        color="black",
        marker="X",
        s=85,
        zorder=4,
        label="BLVPY",
    )
    axes.set(
        xlabel="$C$",
        ylabel="Validation squared-hinge loss",
        xlim=(1e-3, 100.0),
    )
    axes.grid(alpha=0.25)
    axes.legend(frameon=False, fontsize=12)

    figure.tight_layout()
    figure_path = figure_dir / "iris_svm_regularization.pdf"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
