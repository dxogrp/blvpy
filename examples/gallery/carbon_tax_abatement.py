import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Carbon-Tax Design with Sectoral Abatement

    A regulator wants cement, steel, and chemical producers to reduce their
    combined emissions, but each sector chooses its own cost-minimizing
    abatement after seeing the carbon tax. This creates a bilevel policy
    problem: the regulator sets one tax, anticipates the three sector
    responses, and balances an emissions target against the burden of a high
    tax rate.
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
    ## Bilevel policy model

    Let $\tau$ be the carbon tax and let $a_i$ be abatement in sector $i$.
    Sector $i$ starts with baseline emissions $e_i$ and incurs the convex
    abatement cost

    \[
    C_i(a_i)=c_i\left(e^{k_i a_i}-1\right).
    \]

    For a fixed tax, the sectors jointly solve the separable lower problem

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_a
      & \displaystyle\sum_i C_i(a_i)
        +\tau\sum_i(e_i-a_i)\\
    \mathop{\mathrm{subject\ to}}
      & 0\preceq a\preceq e.
    \end{array}
    \]

    The tax term charges each sector for its residual emissions. The
    regulator anticipates the sectors' unique response and solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{\tau,a}
      & \left(\sum_i(e_i-a_i)-2\right)^2+0.01\tau^2\\
    \mathop{\mathrm{subject\ to}}
      & 0.05\leq\tau\leq3,\qquad
        a\in S(\tau).
    \end{array}
    \]

    Exact `cp.exp` atoms in the lower objective canonicalize to three
    exponential-cone blocks, one for each sector.
    """)
    return


@app.cell
def _(np):
    sector_names = np.array(["Cement", "Steel", "Chemicals"])
    baseline_emissions = np.array([2.0, 1.5, 1.0])
    abatement_cost = np.array([0.20, 0.35, 0.50])
    abatement_curvature = np.array([1.0, 1.2, 0.8])

    emissions_target = 2.0
    tax_penalty = 0.01
    tax_bounds = (0.05, 3.0)
    return (
        abatement_cost,
        abatement_curvature,
        baseline_emissions,
        emissions_target,
        sector_names,
        tax_bounds,
        tax_penalty,
    )


@app.cell(hide_code=True)
def _(
    abatement_cost,
    abatement_curvature,
    baseline_emissions,
    mo,
    sector_names,
):
    _rows = "\n".join(
        f"| {name} | {emissions:.2f} | {cost:.2f} | {curvature:.2f} |"
        for name, emissions, cost, curvature in zip(
            sector_names,
            baseline_emissions,
            abatement_cost,
            abatement_curvature,
            strict=True,
        )
    )
    mo.md(f"""
    ## Stylized sector data

    All quantities use normalized units. A larger $c_i$ raises initial
    abatement cost, while a larger $k_i$ makes marginal cost grow faster.

    | sector | baseline emissions $e_i$ | cost scale $c_i$ | curvature $k_i$ |
    | --- | ---: | ---: | ---: |
    {_rows}
    """)
    return


@app.cell
def _(
    abatement_cost,
    abatement_curvature,
    baseline_emissions,
    emissions_target,
    np,
    tax_penalty,
):
    def closed_form_abatement(tax):
        tax_array = np.asarray(tax, dtype=float)
        unconstrained = np.log(tax_array[..., None] / (abatement_cost * abatement_curvature)) / abatement_curvature
        return np.clip(unconstrained, 0.0, baseline_emissions)

    def regulator_cost(tax):
        response = closed_form_abatement(tax)
        residual_emissions = np.sum(baseline_emissions - response, axis=-1)
        return np.square(residual_emissions - emissions_target) + tax_penalty * np.square(tax)

    return closed_form_abatement, regulator_cost


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Closed-form sector response

    In an interior sector, first-order optimality gives

    \[
    c_i k_i e^{k_i a_i}=\tau.
    \]

    Applying the physical bounds therefore gives the exact response

    \[
    a_i(\tau)=\operatorname{clip}\!\left(
      \frac{\log(\tau/(c_i k_i))}{k_i},\,0,\,e_i
    \right).
    \]

    We use this formula only as an independent numerical check; the bilevel
    model below is built and solved from its CVXPY expressions.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    abatement_cost,
    abatement_curvature,
    baseline_emissions,
    closed_form_abatement,
    cp,
    emissions_target,
    tax_bounds,
    tax_penalty,
):
    carbon_tax = cp.Variable(name="carbon_tax")
    abatement = cp.Variable(3, name="abatement")

    carbon_tax.value = 0.75
    abatement.value = closed_form_abatement(float(carbon_tax.value))

    _sector_cost = cp.sum(
        cp.multiply(
            abatement_cost,
            cp.exp(cp.multiply(abatement_curvature, abatement)) - 1.0,
        )
    )
    _tax_payment = carbon_tax * cp.sum(baseline_emissions - abatement)
    _sector_problem = LowerProblem(
        cp.Minimize(_sector_cost + _tax_payment),
        [abatement >= 0.0, abatement <= baseline_emissions],
        parameters=[carbon_tax],
    )
    problem = BilevelProblem(
        cp.Minimize(
            cp.square(cp.sum(baseline_emissions - abatement) - emissions_target) + tax_penalty * cp.square(carbon_tax)
        ),
        _sector_problem,
        upper_constraints=[
            carbon_tax >= tax_bounds[0],
            carbon_tax <= tax_bounds[1],
        ],
    )

    _canonical = problem.canonicalize()
    cone_layout = _canonical.cone_layout
    assert cone_layout.exponential == 3
    assert cone_layout.second_order == ()
    assert cone_layout.power_3d == ()
    return abatement, carbon_tax, problem


@app.cell
def _(closed_form_abatement, minimize_scalar, regulator_cost, tax_bounds):
    reference = minimize_scalar(
        regulator_cost,
        bounds=tax_bounds,
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert reference.success, reference.message

    reference_tax = float(reference.x)
    reference_abatement = closed_form_abatement(reference_tax)
    reference_objective = float(regulator_cost(reference_tax))
    return reference_abatement, reference_objective, reference_tax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve and independently validate the policy

    We initialize the tax and abatement deterministically, then solve to
    $\epsilon=10^{-5}$. Separately, SciPy minimizes the regulator's scalar
    objective after substituting the closed-form sector response.
    """)
    return


@app.cell
def _(abatement, carbon_tax, cp, np, problem):
    epsilon_target = 1e-5
    result = problem.solve(
        epsilon_initial=1e-3,
        epsilon_target=epsilon_target,
        seed=2026,
        solver=cp.IPOPT,
        solver_options={
            "hessian_approximation": "limited-memory",
            "tol": 1e-8,
        },
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)

    result_tax = float(np.asarray(result.variable_values[carbon_tax]))
    result_abatement = np.asarray(result.variable_values[abatement], dtype=float)
    return diagnostics, epsilon_target, result, result_abatement, result_tax


@app.cell(hide_code=True)
def _(
    baseline_emissions,
    closed_form_abatement,
    diagnostics,
    emissions_target,
    epsilon_target,
    mo,
    np,
    reference_abatement,
    reference_objective,
    reference_tax,
    result,
    result_abatement,
    result_tax,
    sector_names,
):
    _closed_form_at_result = closed_form_abatement(result_tax)
    _residual_emissions = baseline_emissions - result_abatement
    _total_residual = float(np.sum(_residual_emissions))

    assert result.succeeded, result.message
    assert diagnostics.source_gap is not None
    np.testing.assert_allclose(result.final_epsilon, epsilon_target, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result_abatement, _closed_form_at_result, atol=5e-3, rtol=0.0)
    np.testing.assert_allclose(result_tax, reference_tax, atol=3e-3, rtol=0.0)
    np.testing.assert_allclose(result_abatement, reference_abatement, atol=5e-3, rtol=0.0)
    np.testing.assert_allclose(result.objective, reference_objective, atol=1e-3, rtol=0.0)
    assert result.residuals.max_violation <= 1e-5
    assert result.complementarity <= 1.1 * epsilon_target
    assert -1e-7 <= diagnostics.source_gap <= 1.1 * epsilon_target

    _response_rows = "\n".join(
        f"| {name} | {abatement_value:.4f} | {residual_value:.4f} |"
        for name, abatement_value, residual_value in zip(
            sector_names,
            result_abatement,
            _residual_emissions,
            strict=True,
        )
    )
    mo.md(rf"""
    ## Result

    | quantity | BLVPY | independent reference |
    | --- | ---: | ---: |
    | status | `{result.status}` | `bounded scalar optimum` |
    | carbon tax $\tau$ | {result_tax:.6f} | {reference_tax:.6f} |
    | total residual emissions | {_total_residual:.6f} | {float(np.sum(baseline_emissions - reference_abatement)):.6f} |
    | emissions target | {emissions_target:.6f} | {emissions_target:.6f} |
    | regulator objective | {float(result.objective):.6f} | {reference_objective:.6f} |

    | sector | abatement $a_i$ | residual emissions $e_i-a_i$ |
    | --- | ---: | ---: |
    {_response_rows}

    | numerical diagnostic | value |
    | --- | ---: |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | primal cone distance | {result.residuals.primal_cone:.3e} |
    | dual cone distance | {result.residuals.dual_cone:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The chosen tax brings aggregate residual emissions close to the target.
    The small remaining difference reflects the explicit tax penalty: forcing
    emissions exactly to two would require a slightly higher tax.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the tax changes sector emissions

    The left panel traces the exact sector responses across the admissible
    tax range. Different cost scales and curvatures produce different
    abatement rates even though every sector faces the same tax. The right
    panel compares each sector's baseline with its residual emissions at the
    selected tax.
    """)
    return


@app.cell
def _(
    baseline_emissions,
    closed_form_abatement,
    np,
    result_abatement,
    tax_bounds,
):
    tax_grid = np.linspace(tax_bounds[0], tax_bounds[1], 500)
    abatement_grid = closed_form_abatement(tax_grid)
    residual_grid = baseline_emissions - abatement_grid
    total_residual_grid = np.sum(residual_grid, axis=1)
    selected_residual_emissions = baseline_emissions - result_abatement
    return (
        residual_grid,
        selected_residual_emissions,
        tax_grid,
        total_residual_grid,
    )


@app.cell(hide_code=True)
def _(
    baseline_emissions,
    emissions_target,
    np,
    plt,
    reference_tax,
    residual_grid,
    result_tax,
    sector_names,
    selected_residual_emissions,
    tax_grid,
    total_residual_grid,
):
    fig, (response_axis, sector_axis) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    for _index, _name in enumerate(sector_names):
        response_axis.plot(
            tax_grid,
            residual_grid[:, _index],
            label=_name,
            linewidth=1.5,
        )
    response_axis.plot(
        tax_grid,
        total_residual_grid,
        color="k",
        linewidth=2.2,
        label="Total",
    )
    response_axis.axhline(
        emissions_target,
        color="grey",
        linestyle=":",
        label="Target",
    )
    response_axis.axvline(result_tax, color="C3", linestyle="--", label="BLVPY tax")
    response_axis.axvline(reference_tax, color="C3", linestyle=":", label="Reference tax")
    response_axis.set(
        xlabel=r"Carbon tax $\tau$",
        ylabel="Residual emissions",
        xlim=(tax_grid[0], tax_grid[-1]),
        ylim=(0.0, 4.7),
    )
    response_axis.legend(frameon=False, fontsize=10, ncol=2)

    _positions = np.arange(sector_names.size)
    _width = 0.36
    sector_axis.bar(
        _positions - _width / 2,
        baseline_emissions,
        width=_width,
        color="grey",
        alpha=0.75,
        label="Baseline",
    )
    sector_axis.bar(
        _positions + _width / 2,
        selected_residual_emissions,
        width=_width,
        color="C2",
        label="After abatement",
    )
    sector_axis.set(
        ylabel="Normalized emissions",
        xticks=_positions,
        xticklabels=sector_names,
        ylim=(0.0, 2.2),
    )
    sector_axis.legend(frameon=False, fontsize=11)

    fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
