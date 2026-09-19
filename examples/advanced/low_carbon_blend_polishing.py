import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gate a low-carbon blend policy with polishing

    A public buyer wants to encourage a lower-carbon binder, but the producer
    retains control of the material recipe. The buyer chooses six rebate
    rates, and the producer responds with a minimum-cost blend subject to
    strength, durability, and availability limits.

    This example uses `polish()` as a deployment gate. A first continuation
    solve looks attractive, but its exact fixed-rebate response misses a 1%
    objective-degradation threshold. We tighten the continuation target,
    polish again, and explicitly adopt only the second candidate.
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
    ## Buyer and producer decisions

    Let $r\in\mathbf{R}^6$ be the buyer's rebate rates and
    $y\in\mathbf{R}^6$ the producer's material shares. For a fixed rebate,
    the producer solves the strongly convex quadratic program

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_y
      & (c-r)^T y+\rho\lVert y-y_{\mathrm{ref}}\rVert_2^2 \\
    \mathop{\mathrm{subject\ to}}
      & \mathbf{1}^T y=1,\\
      & s^T y\geq0.84,\qquad d^T y\geq0.82,\\
      & 0\preceq y\preceq u.
    \end{array}
    \]

    Here $c$ is base cost, $s$ and $d$ are normalized quality scores, $u$
    contains material-specific availability limits, and the positive
    regularizer $\rho$ makes the response unique. Anticipating that response,
    the buyer solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{r,y}
      & e^T y+1.5\lVert r\rVert_2^2 \\
    \mathop{\mathrm{subject\ to}}
      & 0\preceq r\preceq0.08,\qquad \mathbf{1}^T r\leq0.03,\\
      & y\text{ solves the producer problem for }r,
    \end{array}
    \]

    where $e^T y$ is normalized carbon intensity. The sum constraint is an
    aggregate cap on posted rates, not a model of realized rebate spending.
    The quadratic rebate term discourages a large or needlessly concentrated
    policy.
    """)
    return


@app.cell
def _(np):
    material_names = np.array(
        [
            "Clinker",
            "Limestone",
            "Calcined clay",
            "Recycled glass",
            "Fly ash",
            "Slag",
        ]
    )
    base_cost = np.array([0.105, 0.088, 0.072, 0.066, 0.060, 0.082])
    carbon_intensity = np.array([0.92, 0.48, 0.18, 0.08, 0.12, 0.28])
    strength_score = np.array([1.20, 1.02, 0.82, 0.62, 0.70, 0.92])
    durability_score = np.array([0.92, 0.88, 0.82, 0.70, 0.76, 0.95])
    reference_blend = np.array([0.46, 0.18, 0.12, 0.08, 0.08, 0.08])
    material_capacity = np.array([0.65, 0.45, 0.35, 0.28, 0.30, 0.35])

    n_materials = material_names.size
    producer_regularization = 0.025
    strength_minimum = 0.84
    durability_minimum = 0.82
    aggregate_rebate_cap = 0.03
    rebate_penalty = 1.5
    return (
        aggregate_rebate_cap,
        base_cost,
        carbon_intensity,
        durability_minimum,
        durability_score,
        material_capacity,
        material_names,
        n_materials,
        producer_regularization,
        rebate_penalty,
        reference_blend,
        strength_minimum,
        strength_score,
    )


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    aggregate_rebate_cap,
    base_cost,
    carbon_intensity,
    cp,
    durability_minimum,
    durability_score,
    material_capacity,
    n_materials,
    np,
    producer_regularization,
    rebate_penalty,
    reference_blend,
    strength_minimum,
    strength_score,
):
    rebate = cp.Variable(
        n_materials,
        nonneg=True,
        bounds=[0.0, 0.08],
        name="rebate",
    )
    blend = cp.Variable(n_materials, nonneg=True, name="blend")

    rebate.value = np.full(n_materials, aggregate_rebate_cap / n_materials)
    blend.value = reference_blend.copy()

    producer = LowerProblem(
        cp.Minimize((base_cost - rebate) @ blend + producer_regularization * cp.sum_squares(blend - reference_blend)),
        [
            cp.sum(blend) == 1.0,
            strength_score @ blend >= strength_minimum,
            durability_score @ blend >= durability_minimum,
            blend <= material_capacity,
        ],
        parameters=[rebate],
    )
    problem = BilevelProblem(
        cp.Minimize(carbon_intensity @ blend + rebate_penalty * cp.sum_squares(rebate)),
        producer,
        upper_constraints=[cp.sum(rebate) <= aggregate_rebate_cap],
    )
    return blend, problem, rebate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exact no-policy baseline

    Before solving the bilevel policy, we independently solve the producer's
    convex problem with all rebates fixed to zero. This gives a reproducible
    implementation baseline.
    """)
    return


@app.cell
def _(
    base_cost,
    carbon_intensity,
    cp,
    durability_minimum,
    durability_score,
    material_capacity,
    n_materials,
    np,
    producer_regularization,
    reference_blend,
    strength_minimum,
    strength_score,
):
    _baseline_variable = cp.Variable(n_materials, nonneg=True, name="baseline_blend")
    _baseline_problem = cp.Problem(
        cp.Minimize(
            base_cost @ _baseline_variable
            + producer_regularization * cp.sum_squares(_baseline_variable - reference_blend)
        ),
        [
            cp.sum(_baseline_variable) == 1.0,
            strength_score @ _baseline_variable >= strength_minimum,
            durability_score @ _baseline_variable >= durability_minimum,
            _baseline_variable <= material_capacity,
        ],
    )
    _baseline_problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    assert _baseline_problem.status == cp.OPTIMAL

    baseline_blend = np.asarray(_baseline_variable.value, dtype=float)
    baseline_carbon = float(carbon_intensity @ baseline_blend)
    np.testing.assert_allclose(np.sum(baseline_blend), 1.0, atol=1e-9, rtol=0.0)
    assert baseline_carbon > 0.28
    return baseline_blend, baseline_carbon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First pass: a practical continuation target

    We first stop continuation at $\epsilon=10^{-4}$. The result passes
    BLVPY's relaxed residual checks, but the permitted lower-objective gap is
    large enough on this normalized scale to make the returned blend look
    more favorable to the buyer than the producer's exact response.
    """)
    return


@app.cell
def _(blend, cp, np, problem, rebate):
    coarse_epsilon = 1e-4
    coarse_result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=coarse_epsilon,
        solver=cp.IPOPT,
        verbose=True,
    )

    assert coarse_result.succeeded, coarse_result.message
    assert coarse_result.objective is not None
    np.testing.assert_allclose(coarse_result.final_epsilon, coarse_epsilon, atol=0.0, rtol=0.0)

    coarse_result_rebate = np.asarray(coarse_result.variable_values[rebate], dtype=float)
    coarse_result_blend = np.asarray(coarse_result.variable_values[blend], dtype=float)
    coarse_live_state_before_polish = {
        _variable: np.array(_variable.value, dtype=float, copy=True) for _variable in coarse_result.variable_values
    }
    coarse_result_snapshots = {
        _variable: np.array(_value, dtype=float, copy=True)
        for _variable, _value in coarse_result.variable_values.items()
    }
    assert all(not np.asarray(_value).flags.writeable for _value in coarse_result.variable_values.values())
    return (
        coarse_epsilon,
        coarse_live_state_before_polish,
        coarse_result,
        coarse_result_blend,
        coarse_result_rebate,
        coarse_result_snapshots,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Polish, measure, and reject

    `polish()` fixes the six rebate rates from the result snapshot and solves
    the constrained producer problem with no continuation allowance. The
    policy team uses a deliberately explicit rule here: the polished point
    must be feasible, and its upper objective may degrade by at most 1%
    relative to the epsilon-relaxed result.
    """)
    return


@app.cell
def _(
    base_cost,
    blend,
    carbon_intensity,
    coarse_live_state_before_polish,
    coarse_result,
    coarse_result_blend,
    coarse_result_rebate,
    coarse_result_snapshots,
    cp,
    np,
    problem,
    producer_regularization,
    rebate,
    reference_blend,
):
    coarse_polished = problem.polish(
        coarse_result,
        solver=cp.CLARABEL,
        verbose=True,
    )

    coarse_polished_rebate = np.asarray(coarse_polished.variable_values[rebate], dtype=float)
    coarse_polished_blend = np.asarray(coarse_polished.variable_values[blend], dtype=float)
    coarse_result_carbon = float(carbon_intensity @ coarse_result_blend)
    coarse_polished_carbon = float(carbon_intensity @ coarse_polished_blend)

    coarse_result_lower_value = float(
        (base_cost - coarse_result_rebate) @ coarse_result_blend
        + producer_regularization * np.sum(np.square(coarse_result_blend - reference_blend))
    )
    coarse_polished_lower_value = float(
        (base_cost - coarse_polished_rebate) @ coarse_polished_blend
        + producer_regularization * np.sum(np.square(coarse_polished_blend - reference_blend))
    )
    coarse_lower_gap = coarse_result_lower_value - coarse_polished_lower_value

    maximum_degradation = 0.01
    assert coarse_polished.objective_improvement_ratio is not None
    coarse_degradation = -coarse_polished.objective_improvement_ratio
    coarse_accepted = coarse_polished.feasible and coarse_degradation <= maximum_degradation

    assert coarse_polished.feasible
    assert 0.05 < coarse_degradation < 0.08
    assert not coarse_accepted
    assert 5e-5 < coarse_lower_gap < 1.5e-4
    assert coarse_polished_carbon - coarse_result_carbon > 0.01
    assert coarse_polished_lower_value < coarse_result_lower_value
    np.testing.assert_allclose(coarse_polished_rebate, coarse_result_rebate, atol=1e-12, rtol=0.0)

    for _variable, _value in coarse_live_state_before_polish.items():
        np.testing.assert_array_equal(np.asarray(_variable.value), _value)
    for _variable, _value in coarse_result_snapshots.items():
        np.testing.assert_array_equal(coarse_result.variable_values[_variable], _value)
    assert all(not np.asarray(_value).flags.writeable for _value in coarse_polished.variable_values.values())

    coarse_polished_snapshots = {
        _variable: np.array(_value, dtype=float, copy=True)
        for _variable, _value in coarse_polished.variable_values.items()
    }
    return (
        coarse_accepted,
        coarse_degradation,
        coarse_lower_gap,
        coarse_polished,
        coarse_polished_blend,
        coarse_polished_carbon,
        coarse_polished_snapshots,
        coarse_result_carbon,
        maximum_degradation,
    )


@app.cell(hide_code=True)
def _(
    coarse_accepted,
    coarse_degradation,
    coarse_epsilon,
    coarse_lower_gap,
    coarse_polished,
    coarse_polished_carbon,
    coarse_result,
    coarse_result_carbon,
    maximum_degradation,
    mo,
):
    _decision = "accept" if coarse_accepted else "reject and refine"
    mo.md(f"""
    ## First decision: {_decision}

    | Candidate | Carbon intensity | Upper objective |
    |---|---:|---:|
    | $\\epsilon={coarse_epsilon:.0e}$ solve | {coarse_result_carbon:.6f} | {coarse_result.objective:.6f} |
    | Fixed-rebate polish | {coarse_polished_carbon:.6f} | {coarse_polished.objective:.6f} |

    The producer's objective improves by `{coarse_lower_gap:.2e}`, almost the
    full continuation allowance. That small lower-level correction raises the
    buyer's objective by `{100.0 * coarse_degradation:.2f}%`, well beyond the
    illustrative `{100.0 * maximum_degradation:.1f}%` gate. The polished
    candidate is feasible, but feasibility alone is not the acceptance rule.

    We therefore leave the polished snapshots unassigned and rerun the
    continuation with a tighter target.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Second pass: tighten, then polish again

    The model is unchanged, and we reset its live variables to the same
    neutral initialization used for the first pass. Only the continuation
    target moves from $10^{-4}$ to $10^{-6}$. The earlier `BilevelResult` and
    `PolishResult` remain immutable snapshots even though the new solve
    updates the live CVXPY variables.
    """)
    return


@app.cell
def _(
    aggregate_rebate_cap,
    blend,
    coarse_polished,
    coarse_polished_snapshots,
    coarse_result,
    coarse_result_snapshots,
    cp,
    np,
    problem,
    rebate,
    reference_blend,
):
    refined_epsilon = 1e-6
    rebate.project_and_assign(np.full(rebate.shape, aggregate_rebate_cap / rebate.size))
    blend.project_and_assign(reference_blend)
    refined_result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=refined_epsilon,
        solver=cp.IPOPT,
        verbose=True,
    )

    assert refined_result.succeeded, refined_result.message
    assert refined_result.objective is not None
    np.testing.assert_allclose(refined_result.final_epsilon, refined_epsilon, atol=0.0, rtol=0.0)

    for _variable, _value in coarse_result_snapshots.items():
        np.testing.assert_array_equal(coarse_result.variable_values[_variable], _value)
    for _variable, _value in coarse_polished_snapshots.items():
        np.testing.assert_array_equal(coarse_polished.variable_values[_variable], _value)

    refined_result_rebate = np.asarray(refined_result.variable_values[rebate], dtype=float)
    refined_result_blend = np.asarray(refined_result.variable_values[blend], dtype=float)
    refined_live_state_before_polish = {
        _variable: np.array(_variable.value, dtype=float, copy=True) for _variable in refined_result.variable_values
    }
    refined_result_snapshots = {
        _variable: np.array(_value, dtype=float, copy=True)
        for _variable, _value in refined_result.variable_values.items()
    }
    return (
        refined_epsilon,
        refined_live_state_before_polish,
        refined_result,
        refined_result_blend,
        refined_result_rebate,
        refined_result_snapshots,
    )


@app.cell
def _(
    base_cost,
    baseline_carbon,
    blend,
    carbon_intensity,
    coarse_degradation,
    cp,
    maximum_degradation,
    np,
    problem,
    producer_regularization,
    rebate,
    reference_blend,
    refined_live_state_before_polish,
    refined_result,
    refined_result_blend,
    refined_result_rebate,
    refined_result_snapshots,
):
    refined_polished = problem.polish(
        refined_result,
        solver=cp.CLARABEL,
        verbose=True,
    )

    refined_polished_rebate = np.asarray(refined_polished.variable_values[rebate], dtype=float)
    refined_polished_blend = np.asarray(refined_polished.variable_values[blend], dtype=float)
    refined_result_carbon = float(carbon_intensity @ refined_result_blend)
    refined_polished_carbon = float(carbon_intensity @ refined_polished_blend)

    refined_result_lower_value = float(
        (base_cost - refined_result_rebate) @ refined_result_blend
        + producer_regularization * np.sum(np.square(refined_result_blend - reference_blend))
    )
    refined_polished_lower_value = float(
        (base_cost - refined_polished_rebate) @ refined_polished_blend
        + producer_regularization * np.sum(np.square(refined_polished_blend - reference_blend))
    )
    refined_lower_gap = refined_result_lower_value - refined_polished_lower_value

    assert refined_polished.objective_improvement_ratio is not None
    refined_degradation = -refined_polished.objective_improvement_ratio
    refined_accepted = refined_polished.feasible and refined_degradation <= maximum_degradation

    assert refined_polished.feasible
    assert refined_accepted
    assert 0.0 < refined_degradation < maximum_degradation
    assert refined_degradation < coarse_degradation
    assert 5e-7 < refined_lower_gap < 1.5e-6
    assert refined_polished_carbon < 0.85 * baseline_carbon
    assert refined_polished_lower_value < refined_result_lower_value
    np.testing.assert_allclose(refined_polished_rebate, refined_result_rebate, atol=1e-12, rtol=0.0)

    for _variable, _value in refined_live_state_before_polish.items():
        np.testing.assert_array_equal(np.asarray(_variable.value), _value)
    for _variable, _value in refined_result_snapshots.items():
        np.testing.assert_array_equal(refined_result.variable_values[_variable], _value)
    assert all(not np.asarray(_value).flags.writeable for _value in refined_polished.variable_values.values())

    refined_polished_snapshots = {
        _variable: np.array(_value, dtype=float, copy=True)
        for _variable, _value in refined_polished.variable_values.items()
    }
    return (
        refined_accepted,
        refined_degradation,
        refined_lower_gap,
        refined_polished,
        refined_polished_blend,
        refined_polished_carbon,
        refined_polished_snapshots,
        refined_result_carbon,
    )


@app.cell(hide_code=True)
def _(
    baseline_carbon,
    coarse_degradation,
    coarse_epsilon,
    coarse_lower_gap,
    coarse_polished,
    coarse_polished_carbon,
    coarse_result,
    coarse_result_carbon,
    maximum_degradation,
    mo,
    refined_accepted,
    refined_degradation,
    refined_epsilon,
    refined_lower_gap,
    refined_polished,
    refined_polished_carbon,
    refined_result,
    refined_result_carbon,
):
    _decision = "accept" if refined_accepted else "reject"
    _coarse_result_row = (
        f"| $\\epsilon={coarse_epsilon:.0e}$ | Relaxed result | "
        f"{coarse_result_carbon:.6f} | {coarse_result.objective:.6f} | -- |"
    )
    _coarse_polished_row = (
        f"| $\\epsilon={coarse_epsilon:.0e}$ | Polish | "
        f"{coarse_polished_carbon:.6f} | {coarse_polished.objective:.6f} | "
        f"{100.0 * coarse_degradation:.2f}% |"
    )
    _refined_result_row = (
        f"| $\\epsilon={refined_epsilon:.0e}$ | Relaxed result | "
        f"{refined_result_carbon:.6f} | {refined_result.objective:.6f} | -- |"
    )
    _refined_polished_row = (
        f"| $\\epsilon={refined_epsilon:.0e}$ | Polish | "
        f"{refined_polished_carbon:.6f} | {refined_polished.objective:.6f} | "
        f"{100.0 * refined_degradation:.2f}% |"
    )
    mo.md(f"""
    ## The tighter candidate passes the gate

    | Pass | Candidate | Carbon intensity | Upper objective | Polish degradation |
    |---|---|---:|---:|---:|
    | Baseline | Zero-rebate response | {baseline_carbon:.6f} | {baseline_carbon:.6f} | -- |
    {_coarse_result_row}
    {_coarse_polished_row}
    {_refined_result_row}
    {_refined_polished_row}

    Tightening epsilon reduces the fixed-rebate lower gap from
    `{coarse_lower_gap:.2e}` to `{refined_lower_gap:.2e}`. The corresponding
    upper-objective degradation falls from `{100.0 * coarse_degradation:.2f}%`
    to `{100.0 * refined_degradation:.2f}%`, below the
    `{100.0 * maximum_degradation:.1f}%` gate. Decision: **{_decision}**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See what polishing changes

    The composition chart shows the larger movement after the coarse solve
    and the much smaller correction after refinement. The decision chart
    compares both objective degradations with the same 1% acceptance rule.
    """)
    return


@app.cell
def _(
    baseline_blend,
    coarse_degradation,
    coarse_polished_blend,
    coarse_result_blend,
    material_names,
    maximum_degradation,
    np,
    plt,
    refined_degradation,
    refined_polished_blend,
    refined_result_blend,
):
    _candidate_labels = [
        "Zero-rebate baseline",
        r"$10^{-4}$ solve",
        r"$10^{-4}$ polish",
        r"$10^{-6}$ solve",
        r"$10^{-6}$ polish",
    ]
    _candidate_blends = np.vstack(
        [
            baseline_blend,
            coarse_result_blend,
            coarse_polished_blend,
            refined_result_blend,
            refined_polished_blend,
        ]
    )

    _figure, (_composition_axis, _decision_axis) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    _left = np.zeros(len(_candidate_labels))
    _colors = plt.colormaps["tab20c"](np.linspace(0.05, 0.9, len(material_names)))
    for _index, (_material, _color) in enumerate(zip(material_names, _colors, strict=True)):
        _shares = _candidate_blends[:, _index]
        _composition_axis.barh(
            _candidate_labels,
            _shares,
            left=_left,
            color=_color,
            label=_material,
        )
        _left += _shares
    _composition_axis.set_xlim(0.0, 1.0)
    _composition_axis.set_xlabel("Material share")
    _composition_axis.invert_yaxis()
    _composition_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=2,
    )

    _degradations = 100.0 * np.array([coarse_degradation, refined_degradation])
    _decision_axis.bar(
        [r"$10^{-4}$", r"$10^{-6}$"],
        _degradations,
        color=["tab:orange", "tab:green"],
        width=0.58,
    )
    _decision_axis.axhline(
        100.0 * maximum_degradation,
        color="0.25",
        linestyle="--",
        linewidth=1.5,
        label="1% acceptance gate",
    )
    for _index, _value in enumerate(_degradations):
        _decision_axis.text(_index, _value + 0.12, f"{_value:.2f}%", ha="center", va="bottom")
    _decision_axis.set_ylabel("Objective degradation after polish (%)")
    _decision_axis.set_xlabel(r"Continuation target $\epsilon$")
    _decision_axis.legend(frameon=False)

    _figure.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adopt only the accepted snapshot

    `polish()` has not changed the live model. The assignment below is a
    separate, explicit action, guarded by both feasibility and the 1%
    degradation rule. Every original CVXPY variable is assigned so the model
    receives one complete candidate rather than a mixture of result states.
    """)
    return


@app.cell
def _(
    baseline_carbon,
    blend,
    carbon_intensity,
    np,
    rebate,
    refined_accepted,
    refined_polished,
    refined_polished_snapshots,
):
    if not refined_accepted:
        raise RuntimeError("The refined polished candidate did not pass the deployment gate.")

    for _variable, _value in refined_polished.variable_values.items():
        _variable.project_and_assign(_value)
    for _variable, _value in refined_polished_snapshots.items():
        np.testing.assert_array_equal(np.asarray(_variable.value), _value)
        np.testing.assert_array_equal(refined_polished.variable_values[_variable], _value)

    adopted_rebate = np.asarray(rebate.value, dtype=float)
    adopted_blend = np.asarray(blend.value, dtype=float)
    adopted_carbon = float(carbon_intensity @ adopted_blend)
    carbon_reduction = (baseline_carbon - adopted_carbon) / baseline_carbon
    assert carbon_reduction > 0.15
    return adopted_blend, adopted_carbon, adopted_rebate, carbon_reduction


@app.cell(hide_code=True)
def _(
    adopted_blend,
    adopted_carbon,
    adopted_rebate,
    carbon_reduction,
    material_names,
    mo,
):
    _rebates = ", ".join(
        f"{_material}: {_value:.5f}" for _material, _value in zip(material_names, adopted_rebate, strict=True)
    )
    _blend = ", ".join(
        f"{_material}: {_value:.3f}" for _material, _value in zip(material_names, adopted_blend, strict=True)
    )
    mo.md(f"""
    ## Adopted policy response

    - Normalized carbon intensity: **{adopted_carbon:.6f}**
    - Reduction from the exact zero-rebate baseline:
      **{100.0 * carbon_reduction:.1f}%**
    - Rebate rates: `{_rebates}`
    - Producer blend: `{_blend}`

    The model contains the refined polished values only because the guarded
    assignment loop ran. Both bilevel results and both polished results remain
    immutable records of the decisions that led here.
    """)
    return


if __name__ == "__main__":
    app.run()
