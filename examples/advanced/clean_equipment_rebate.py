import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clean-equipment rebates under Cobb--Douglas production

    A regulator wants a producer to substitute clean equipment for a fossil
    input without prescribing the producer's operating plan. The regulator
    therefore chooses a per-unit clean-equipment rebate, and the producer
    responds by purchasing the least-cost input mix that meets a fixed output
    requirement. The Cobb--Douglas production requirement is represented by a
    three-dimensional power cone.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import minimize_scalar

    from blvpy import BilevelProblem, LowerProblem

    _style_path = Path(__file__).resolve().parents[1] / "_shared" / "zhlatex.mplstyle"
    plt.style.use(_style_path)
    return BilevelProblem, LowerProblem, cp, minimize_scalar, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy and producer problems

    Let $r$ be the clean-equipment rebate. The producer chooses clean and
    fossil inputs $(c,f)$ and pays the net operating cost

    \[
      (12-r)c+6f.
    \]

    Normalized output is $c^{0.35}f^{0.65}$, so the producer's response is

    \[
    \begin{array}{ll}
        S(r)=\mathop{\mathrm{argmin}}_{c,f} & (12-r)c+6f\\
        \text{subject to} & c^{0.35}f^{0.65}\geq1.
    \end{array}
    \]

    The regulator targets fossil use of $1.15$ and also accounts for the
    fiscal burden of a larger posted rebate:

    \[
    \begin{array}{ll}
      \mathop{\mathrm{minimize}}_{r,c,f}
        & (f-1.15)^2+0.0005r^2 \\
      \mathop{\mathrm{subject\ to}}
        & 0\leq r\leq8,\\
        & (c,f)\in S(r).
    \end{array}
    \]

    The upper bound on $r$ keeps the net clean-input price positive. Thus the
    producer has a unique finite response throughout the policy interval.
    """)
    return


@app.cell
def _():
    production_elasticity = 0.35
    clean_list_price = 12.0
    fossil_price = 6.0
    fossil_target = 1.15
    rebate_penalty = 0.0005
    rebate_bounds = (0.0, 8.0)
    initial_rebate = 4.0
    return (
        clean_list_price,
        fossil_price,
        fossil_target,
        initial_rebate,
        production_elasticity,
        rebate_bounds,
        rebate_penalty,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Closed-form producer response

    The producer's first-order conditions equate expenditure shares. With
    $\alpha=0.35$ and net clean price $p_c=12-r$, define

    \[
      q(r)=\frac{(1-\alpha)p_c}{\alpha\,6}.
    \]

    At the cost-minimizing production frontier, $f/c=q$ and
    $c^\alpha f^{1-\alpha}=1$. Therefore

    \[
      c^\star(r)=q(r)^{-(1-\alpha)},\qquad
      f^\star(r)=q(r)^\alpha.
    \]

    These formulas provide an independent lower-response check and reduce the
    policy benchmark to a one-dimensional bounded optimization problem.
    """)
    return


@app.cell
def _(
    clean_list_price,
    fossil_price,
    fossil_target,
    np,
    production_elasticity,
    rebate_penalty,
):
    def producer_response(rebate_value):
        _rebate = np.asarray(rebate_value, dtype=float)
        _net_clean_price = clean_list_price - _rebate
        _input_ratio = (1.0 - production_elasticity) * _net_clean_price / (production_elasticity * fossil_price)
        _clean = _input_ratio ** (-(1.0 - production_elasticity))
        _fossil = _input_ratio**production_elasticity
        return _clean, _fossil

    def policy_objective(rebate_value):
        _, _fossil = producer_response(rebate_value)
        return (_fossil - fossil_target) ** 2 + rebate_penalty * np.asarray(rebate_value) ** 2

    return policy_objective, producer_response


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the bilevel power-cone model

    Membership of $(c,f,1)$ in `PowCone3D` with exponent $0.35$ is exactly

    \[
      c\geq0,\qquad f\geq0,\qquad
      1\leq c^{0.35}f^{0.65}.
    \]

    The initial values are deterministic and feasible.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    clean_list_price,
    cp,
    fossil_price,
    fossil_target,
    initial_rebate,
    producer_response,
    production_elasticity,
    rebate_bounds,
    rebate_penalty,
):
    rebate = cp.Variable(bounds=list(rebate_bounds), name="clean_equipment_rebate")
    clean_input = cp.Variable(name="clean_input")
    fossil_input = cp.Variable(name="fossil_input")

    _initial_clean, _initial_fossil = producer_response(initial_rebate)
    rebate.value = initial_rebate
    clean_input.value = float(_initial_clean)
    fossil_input.value = float(_initial_fossil)

    producer = LowerProblem(
        cp.Minimize((clean_list_price - rebate) * clean_input + fossil_price * fossil_input),
        [cp.PowCone3D(clean_input, fossil_input, 1.0, production_elasticity)],
        parameters=[rebate],
    )
    problem = BilevelProblem(
        cp.Minimize(cp.square(fossil_input - fossil_target) + rebate_penalty * cp.square(rebate)),
        producer,
    )
    return clean_input, fossil_input, problem, rebate


@app.cell(hide_code=True)
def _(problem, production_elasticity):
    canonical = problem.canonicalize()
    cone_layout = canonical.cone_layout

    assert cone_layout.power_3d == (production_elasticity,)
    assert len(cone_layout.power_3d_slices) == 1
    assert cone_layout.second_order == ()
    assert cone_layout.exponential == 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent scalar policy benchmark

    Substituting the closed-form producer response into the regulator's
    objective leaves a smooth scalar function on $[0,8]$. SciPy's bounded
    scalar optimizer supplies an independent reference for both the chosen
    rebate and its induced input mix.
    """)
    return


@app.cell
def _(
    minimize_scalar,
    np,
    policy_objective,
    producer_response,
    production_elasticity,
    rebate_bounds,
):
    reference = minimize_scalar(
        policy_objective,
        bounds=rebate_bounds,
        method="bounded",
        options={"xatol": 1e-13},
    )
    assert reference.success, reference.message

    reference_rebate = float(reference.x)
    _reference_clean, _reference_fossil = producer_response(reference_rebate)
    reference_clean = float(_reference_clean)
    reference_fossil = float(_reference_fossil)
    reference_objective = float(reference.fun)

    np.testing.assert_allclose(
        reference_clean**production_elasticity * reference_fossil ** (1.0 - production_elasticity),
        1.0,
        atol=1e-12,
        rtol=0.0,
    )
    return (
        reference_clean,
        reference_fossil,
        reference_objective,
        reference_rebate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve and validate

    BLVPY solves the unreduced bilevel model by continuation to
    $\epsilon=10^{-5}$. We then evaluate the exact producer formula at the
    returned rebate and compare the complete result with the independently
    minimized scalar benchmark.
    """)
    return


@app.cell
def _(
    clean_input,
    cp,
    fossil_input,
    np,
    problem,
    producer_response,
    production_elasticity,
    rebate,
    rebate_bounds,
    reference_clean,
    reference_fossil,
    reference_objective,
    reference_rebate,
):
    epsilon_target = 1e-5
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=epsilon_target,
        feasibility_tolerance=1e-7,
        solver=cp.IPOPT,
        solver_options={"hessian_approximation": "limited-memory", "tol": 1e-8},
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)

    result_rebate = float(np.asarray(result.variable_values[rebate]))
    result_clean = float(np.asarray(result.variable_values[clean_input]))
    result_fossil = float(np.asarray(result.variable_values[fossil_input]))
    _closed_form_clean, _closed_form_fossil = producer_response(result_rebate)
    closed_form_clean = float(_closed_form_clean)
    closed_form_fossil = float(_closed_form_fossil)

    assert result.succeeded, result.message
    assert result.objective is not None
    np.testing.assert_allclose(result.final_epsilon, epsilon_target, atol=0.0, rtol=0.0)
    assert result.residuals.max_violation <= 1e-5
    assert result.complementarity <= 1.1 * epsilon_target
    assert abs(diagnostics.source_gap) <= 1.1 * epsilon_target
    assert rebate_bounds[0] <= result_rebate <= rebate_bounds[1]

    np.testing.assert_allclose(
        [result_clean, result_fossil],
        [closed_form_clean, closed_form_fossil],
        atol=3e-3,
        rtol=0.0,
    )
    np.testing.assert_allclose(result_rebate, reference_rebate, atol=2e-2, rtol=0.0)
    np.testing.assert_allclose(
        [result_clean, result_fossil],
        [reference_clean, reference_fossil],
        atol=3e-3,
        rtol=0.0,
    )
    assert abs(float(result.objective) - reference_objective) <= 5e-3
    assert result_clean**production_elasticity * result_fossil ** (1.0 - production_elasticity) >= 1.0 - 1e-5
    return (
        closed_form_clean,
        closed_form_fossil,
        diagnostics,
        epsilon_target,
        result,
        result_clean,
        result_fossil,
        result_rebate,
    )


@app.cell(hide_code=True)
def _(
    closed_form_clean,
    closed_form_fossil,
    diagnostics,
    epsilon_target,
    mo,
    reference_clean,
    reference_fossil,
    reference_objective,
    reference_rebate,
    result,
    result_clean,
    result_fossil,
    result_rebate,
):
    mo.md(f"""
    ## Result

    | quantity | BLVPY result | independent reference |
    | --- | ---: | ---: |
    | status | `{result.status}` | `success` |
    | clean-equipment rebate | {result_rebate:.6f} | {reference_rebate:.6f} |
    | clean input | {result_clean:.6f} | {reference_clean:.6f} |
    | fossil input | {result_fossil:.6f} | {reference_fossil:.6f} |
    | policy objective | {float(result.objective):.6f} | {reference_objective:.6f} |
    | final epsilon | {result.final_epsilon:.3e} | {epsilon_target:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} | -- |
    | complementarity | {result.complementarity:.3e} | -- |
    | signed source gap | {diagnostics.source_gap:.3e} | -- |

    At the returned rebate, the closed-form producer response is
    $({closed_form_clean:.6f}, {closed_form_fossil:.6f})$. Its agreement with
    the returned lower variables confirms that the policy solution respects
    the producer's cost-minimizing response, while the separate scalar solve
    checks the regulator's decision. The policy objective is shallow near its
    minimum, so the nonzero continuation allowance can leave a small visible
    difference in the rebate even when the input mix and objective agree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy response curves

    The left panel shows how a larger rebate lowers the producer's clean-input
    price, raising clean use and reducing fossil use along the fixed-output
    frontier. The right panel shows the resulting tradeoff between the fossil
    target and the rebate penalty.
    """)
    return


@app.cell
def _(
    np,
    plt,
    policy_objective,
    producer_response,
    rebate_bounds,
    reference_objective,
    reference_rebate,
    result_clean,
    result_fossil,
    result_rebate,
):
    _rebate_grid = np.linspace(rebate_bounds[0], rebate_bounds[1], 240)
    _clean_path, _fossil_path = producer_response(_rebate_grid)
    _policy_path = policy_objective(_rebate_grid)

    _figure, (_input_axis, _policy_axis) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _input_axis.plot(_rebate_grid, _clean_path, linewidth=2.0, label="Clean input")
    _input_axis.plot(_rebate_grid, _fossil_path, linewidth=2.0, label="Fossil input")
    _input_axis.scatter(
        [result_rebate, result_rebate],
        [result_clean, result_fossil],
        color="tab:orange",
        marker="o",
        s=55,
        zorder=3,
        label="BLVPY solution",
    )
    _input_axis.set_xlabel("Clean-equipment rebate")
    _input_axis.set_ylabel("Normalized input use")
    _input_axis.legend(frameon=False)

    _policy_axis.plot(_rebate_grid, _policy_path, color="0.25", linewidth=2.0)
    _policy_axis.scatter(
        [reference_rebate],
        [reference_objective],
        color="tab:green",
        marker="*",
        s=130,
        zorder=4,
        label="Independent optimum",
    )
    _policy_axis.axvline(
        result_rebate,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label="BLVPY rebate",
    )
    _policy_axis.set_xlabel("Clean-equipment rebate")
    _policy_axis.set_ylabel("Policy objective")
    _policy_axis.legend(frameon=False)

    _figure.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
