import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decide whether to adopt a polished ridge fit

    A validation-selected ridge model is a practical bilevel problem: the
    upper problem chooses the regularization penalty, while the lower problem
    fits the regression coefficients. An epsilon-continuation solve permits a
    small amount of lower-level suboptimality, so its validation score can be
    slightly optimistic.

    This example fixes the selected penalty, retrains the coefficients with
    `polish()`, and compares lower-level feasibility with validation
    performance before asking which candidate to place in the live model.
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
    ## Validation selects the ridge penalty

    Let $(X_{\mathrm{tr}},y_{\mathrm{tr}})$ and
    $(X_{\mathrm{val}},y_{\mathrm{val}})$ be training and validation data.
    The upper problem chooses the penalty $\lambda$ by validation error,

    \[
    \mathop{\mathrm{minimize}}_{\lambda,w}\quad
      \frac{1}{m_{\mathrm{val}}}
      \lVert X_{\mathrm{val}}w-y_{\mathrm{val}}\rVert_2^2,
      \qquad 10^{-4}\leq\lambda\leq10,
    \]

    while the coefficients must solve the lower training problem

    \[
    w\in\mathop{\mathrm{argmin}}_v\quad
      \frac{1}{m_{\mathrm{tr}}}
      \lVert X_{\mathrm{tr}}v-y_{\mathrm{tr}}\rVert_2^2
      +\lambda\lVert v\rVert_2^2.
    \]

    The positive penalty makes the lower response unique. We use deterministic
    synthetic data so the numerical tradeoff is reproducible without a data
    download.
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(1907)
    n_features = 8
    n_training = 24
    n_validation = 80

    true_coefficients = np.array([1.6, -1.2, 0.8, 0.0, -0.5, 0.35, 0.0, 0.6])
    X_training = rng.normal(size=(n_training, n_features))
    X_validation = rng.normal(size=(n_validation, n_features))
    feature_scale = np.std(X_training, axis=0)
    X_training = X_training / feature_scale
    X_validation = X_validation / feature_scale
    y_training = X_training @ true_coefficients + rng.normal(scale=1.1, size=n_training)
    y_validation = X_validation @ true_coefficients + rng.normal(scale=0.25, size=n_validation)
    return X_training, X_validation, n_features, n_training, n_validation, y_training, y_validation


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    X_training,
    X_validation,
    cp,
    n_features,
    n_training,
    n_validation,
    y_training,
    y_validation,
):
    penalty = cp.Variable(
        nonneg=True,
        bounds=[1e-4, 10.0],
        name="ridge_penalty",
    )
    coefficients = cp.Variable(n_features, name="coefficients")

    _training_loss = cp.sum_squares(X_training @ coefficients - y_training) / n_training
    lower_problem = LowerProblem(
        cp.Minimize(_training_loss + penalty * cp.sum_squares(coefficients)),
        parameters=[penalty],
    )
    _validation_loss = cp.sum_squares(X_validation @ coefficients - y_validation) / n_validation
    problem = BilevelProblem(cp.Minimize(_validation_loss), lower_problem)
    return coefficients, penalty, problem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve to a practical continuation target

    We retain BLVPY's default
    $\epsilon_{\mathrm{initial}}=10^{-1}$ and continue to
    $\epsilon_{\mathrm{target}}=10^{-4}$. This is a normal solve rather than
    a deliberately one-step or rescaled construction. The returned
    coefficients are valid for the epsilon-relaxed formulation, but they need
    not be the fixed-penalty ridge solution.
    """)
    return


@app.cell
def _(coefficients, cp, np, penalty, problem):
    epsilon_target = 1e-4
    result = problem.solve(
        epsilon_target=epsilon_target,
        solver=cp.IPOPT,
        verbose=False,
    )

    assert result.succeeded, result.message
    assert result.objective is not None
    np.testing.assert_allclose(result.final_epsilon, epsilon_target, atol=0.0, rtol=0.0)

    result_penalty = float(np.asarray(result.variable_values[penalty]))
    result_coefficients = np.asarray(result.variable_values[coefficients], dtype=float)
    result_validation_mse = float(result.objective)
    assert 0.15 < result_penalty < 0.22
    assert 0.82 < result_validation_mse < 0.88

    live_state_before_polish = {
        _variable: np.array(_variable.value, dtype=float, copy=True) for _variable in result.variable_values
    }
    result_snapshots = {
        _variable: np.array(_value, dtype=float, copy=True) for _variable, _value in result.variable_values.items()
    }
    assert all(not np.asarray(_value).flags.writeable for _value in result.variable_values.values())
    return (
        epsilon_target,
        live_state_before_polish,
        result,
        result_coefficients,
        result_penalty,
        result_snapshots,
        result_validation_mse,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrain at the selected penalty

    `polish()` fixes the upper variable at the result snapshot and solves the
    canonical lower problem with zero continuation allowance. Clarabel returns
    a new complete candidate, but `polish()` does not assign it to the CVXPY
    variables.
    """)
    return


@app.cell
def _(
    X_training,
    X_validation,
    coefficients,
    cp,
    live_state_before_polish,
    n_training,
    n_validation,
    np,
    penalty,
    problem,
    result,
    result_coefficients,
    result_penalty,
    result_snapshots,
    result_validation_mse,
    y_training,
    y_validation,
):
    polished = problem.polish(
        result,
        solver=cp.CLARABEL,
        verbose=False,
    )

    polished_penalty = float(np.asarray(polished.variable_values[penalty]))
    polished_coefficients = np.asarray(polished.variable_values[coefficients], dtype=float)
    polished_validation_mse = float(polished.objective)

    result_lower_objective = float(
        np.sum(np.square(X_training @ result_coefficients - y_training)) / n_training
        + result_penalty * np.sum(np.square(result_coefficients))
    )
    polished_lower_objective = float(
        np.sum(np.square(X_training @ polished_coefficients - y_training)) / n_training
        + polished_penalty * np.sum(np.square(polished_coefficients))
    )
    coefficient_change = float(np.linalg.norm(polished_coefficients - result_coefficients))

    _result_validation_check = float(
        np.sum(np.square(X_validation @ result_coefficients - y_validation)) / n_validation
    )
    _polished_validation_check = float(
        np.sum(np.square(X_validation @ polished_coefficients - y_validation)) / n_validation
    )
    np.testing.assert_allclose(result_validation_mse, _result_validation_check, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(polished_validation_mse, _polished_validation_check, atol=1e-10, rtol=0.0)

    assert polished.feasible
    assert polished.objective_improvement_ratio is not None
    assert -0.04 < polished.objective_improvement_ratio < -0.015
    assert polished_validation_mse > result_validation_mse
    assert polished_lower_objective < result_lower_objective
    assert coefficient_change > 1e-3
    np.testing.assert_allclose(polished_penalty, result_penalty, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(polished_validation_mse, 0.87505, atol=0.01, rtol=0.0)

    for _variable, _value in live_state_before_polish.items():
        np.testing.assert_array_equal(np.asarray(_variable.value), _value)
    for _variable, _value in result_snapshots.items():
        np.testing.assert_array_equal(result.variable_values[_variable], _value)
    assert all(not np.asarray(_value).flags.writeable for _value in polished.variable_values.values())

    polished_snapshots = {
        _variable: np.array(_value, dtype=float, copy=True) for _variable, _value in polished.variable_values.items()
    }
    polished_named_values = {
        _variable.name(): np.array2string(np.asarray(_value), precision=4, suppress_small=True)
        for _variable, _value in polished.variable_values.items()
    }
    return (
        coefficient_change,
        polished,
        polished_coefficients,
        polished_lower_objective,
        polished_named_values,
        polished_penalty,
        polished_snapshots,
        polished_validation_mse,
        result_lower_objective,
    )


@app.cell(hide_code=True)
def _(
    coefficient_change,
    epsilon_target,
    mo,
    polished,
    polished_lower_objective,
    polished_named_values,
    polished_penalty,
    polished_validation_mse,
    result_lower_objective,
    result_penalty,
    result_validation_mse,
):
    _snapshots = "<br>".join(f"`{name}`: `{value}`" for name, value in polished_named_values.items())
    _result_row = (
        f"| $\\epsilon={epsilon_target:.0e}$ solve | {result_penalty:.6f} | "
        f"{result_validation_mse:.6f} | {result_lower_objective:.6f} |"
    )
    _polished_row = (
        f"| Polished | {polished_penalty:.6f} | {polished_validation_mse:.6f} | {polished_lower_objective:.6f} |"
    )
    mo.md(f"""
    ## Compare the candidates

    | Candidate | Ridge penalty | Validation MSE | Lower ridge objective |
    |---|---:|---:|---:|
    {_result_row}
    {_polished_row}

    The coefficient vector moves by
    $\\lVert w_{{\\mathrm{{polished}}}}-w_{{\\epsilon}}\\rVert_2
    ={coefficient_change:.5f}$ while the penalty remains fixed. The lower
    objective improves by about
    ${result_lower_objective - polished_lower_objective:.1e}$, but validation
    MSE increases by
    ${100.0 * -polished.objective_improvement_ratio:.1f}\\%$.

    The immutable `PolishResult` exposes all the information needed for the
    decision:

    - `feasible`: `{str(polished.feasible).lower()}`
    - `objective`: `{polished.objective:.6f}`
    - `objective_improvement_ratio`: `{polished.objective_improvement_ratio:.6f}`
    - `variable_values`:<br>{_snapshots}

    The negative ratio means that the polished minimization objective is
    worse. Its feasibility flag nevertheless confirms that the complete
    fixed-penalty candidate passes BLVPY's standard checks. The live CVXPY
    state still contains the epsilon-relaxed result at this point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why the validation score gets worse

    For this unconstrained ridge problem, the fixed-penalty response can also
    be computed from its linear optimality system. The curve below evaluates
    validation MSE along those responses. The polished point lies on that
    path, while the epsilon-relaxed point sits below it at the same selected
    penalty.
    """)
    return


@app.cell
def _(
    X_training,
    X_validation,
    n_features,
    n_training,
    n_validation,
    np,
    plt,
    polished_penalty,
    polished_validation_mse,
    result_penalty,
    result_validation_mse,
    y_training,
    y_validation,
):
    _penalty_grid = np.geomspace(1e-4, 10.0, 160)
    _gram = X_training.T @ X_training / n_training
    _cross_product = X_training.T @ y_training / n_training
    _identity = np.eye(n_features)
    _path_validation_mse = []
    for _penalty in _penalty_grid:
        _path_coefficients = np.linalg.solve(_gram + _penalty * _identity, _cross_product)
        _path_validation_mse.append(np.sum(np.square(X_validation @ _path_coefficients - y_validation)) / n_validation)

    _figure, _axis = plt.subplots(figsize=(7.0, 4.5))
    _axis.semilogx(
        _penalty_grid,
        _path_validation_mse,
        color="0.25",
        linewidth=2.0,
        label="Fixed-response ridge path",
    )
    _axis.scatter(
        [result_penalty],
        [result_validation_mse],
        color="tab:orange",
        marker="s",
        s=75,
        label=r"$\epsilon$-relaxed result",
        zorder=3,
    )
    _axis.scatter(
        [polished_penalty],
        [polished_validation_mse],
        color="tab:green",
        marker="*",
        s=140,
        label="Polished candidate",
        zorder=4,
    )
    _axis.set_xlabel(r"Ridge penalty $\lambda$")
    _axis.set_ylabel("Validation MSE")
    _axis.legend(frameon=False)
    _figure.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decide which candidate to keep

    Feasibility and upper-objective quality answer different questions. The
    polished coefficients are consistent with the fitted ridge model, but
    they cost about 2.7% in validation MSE. Whether that consistency is worth
    the degradation depends on the application.

    Keeping the default selection means knowingly accepting the
    epsilon-relaxed result. If neither candidate is acceptable, rerun the
    bilevel solve with a tighter epsilon target. Changing the selection below
    explicitly assigns every variable from the chosen immutable snapshot.
    """)
    return


@app.cell
def _(mo):
    candidate_choice = mo.ui.radio(
        options={
            "Keep epsilon-relaxed result": "result",
            "Adopt polished candidate": "polished",
        },
        value="Keep epsilon-relaxed result",
        label="Candidate to place in the model",
    )
    candidate_choice
    return (candidate_choice,)


@app.cell
def _(
    candidate_choice,
    np,
    polished,
    polished_snapshots,
    result,
    result_snapshots,
):
    assert candidate_choice.value in {"result", "polished"}
    selected_candidate = polished if candidate_choice.value == "polished" else result
    selected_name = "Polished candidate" if candidate_choice.value == "polished" else "Epsilon-relaxed result"

    for _variable, _value in selected_candidate.variable_values.items():
        _variable.project_and_assign(_value)
    for _variable, _value in selected_candidate.variable_values.items():
        np.testing.assert_array_equal(np.asarray(_variable.value), _value)

    for _variable, _value in result_snapshots.items():
        np.testing.assert_array_equal(result.variable_values[_variable], _value)
    for _variable, _value in polished_snapshots.items():
        np.testing.assert_array_equal(polished.variable_values[_variable], _value)

    selected_objective = float(selected_candidate.objective)
    return selected_name, selected_objective


@app.cell(hide_code=True)
def _(mo, selected_name, selected_objective):
    mo.md(f"""
    ### Current model state

    - Selected snapshot: **{selected_name}**
    - Validation MSE: **{selected_objective:.6f}**

    This state changed only through the explicit `project_and_assign` loop
    above. Toggling the selector assigns the other complete snapshot without
    modifying either result object.
    """)
    return


if __name__ == "__main__":
    app.run()
