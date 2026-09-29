import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Renewable Capacity Planning with German Grid Data

    We consider renewable capacity planning in a power system, where a
    generation planner must choose renewable capacity before an electricity
    dispatcher sees the resulting operating limits and optimizes the dispatch
    accordingly. The planner anticipates that response and trades investment
    cost against the thermal generation that remains in the dispatch.

    The hourly demand and renewable-availability profile comes from observed
    German grid data for 21 June 2019.
    """)
    return


@app.cell
def _():
    import csv
    from datetime import datetime
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from blvpy import BilevelProblem, LowerProblem

    plt.style.use(Path(__file__).resolve().parents[1] / "_shared" / "zhlatex.mplstyle")
    return BilevelProblem, LowerProblem, Path, cp, csv, datetime, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bilevel formulation

    Let $\alpha\in\mathbf{R}$ denote renewable capacity in 10-GW blocks, and
    let $L\in\mathbf{R}^m$ denote the observed aggregate renewable capacity
    factor over $m$ time periods. Thus, $\alpha L\in\mathbf{R}^m$ gives the
    renewable-generation limits, also in 10-GW blocks. Let
    $D\in\mathbf{R}^m$ denote electricity demand in the same units, and let
    $r,g\in\mathbf{R}^m$ denote renewable and thermal dispatch. The generation
    planner solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{\alpha,r,g}
        & p_r\alpha^2+p_g\mathbf{1}^Tg \\
    \mathop{\mathrm{subject\ to}}
        & 0\leq\alpha\leq\alpha_{\mathrm{max}},\\
        & (r,g)\in S(\alpha).
    \end{array}
    \]

    For a fixed $\alpha$, the grid dispatcher solves the lower problem

    \[
    \begin{array}{ll}
    S(\alpha)=\mathop{\mathrm{argmin}}_{r,g}
        & c_r\mathbf{1}^Tr+c_g\mathbf{1}^Tg
          +\delta\left(\lVert r\rVert_2^2+\lVert g\rVert_2^2\right) \\
    \mathop{\mathrm{subject\ to}}
        & 0\preceq r\preceq\alpha L,\quad g\succeq0,\\
        & r+g\succeq D.
    \end{array}
    \]

    The upper variable is $\alpha$, while $r$ and $g$ are the lower variables.
    The planning and dispatch cost coefficients $p_r,p_g,c_r,c_g$, maximum
    capacity $\alpha_{\mathrm{max}}$, regularization parameter $\delta$, and
    vectors $L,D$ are given. The small quadratic term makes the lower response
    stable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observed German operating data

    The 24 rows cover the local day 21 June 2019 in Central European Summer
    Time and include grid load, onshore and offshore wind generation,
    photovoltaics, and their installed capacities. Source:
    **Bundesnetzagentur | SMARD.de**, licensed under
    [CC BY 4.0](https://www.smard.de/en/datennutzung).
    """)
    return


@app.cell
def _(Path, csv, datetime, np):
    data_path = Path(__file__).resolve().parents[1] / "_shared" / "data" / "smard_germany_2019-06-21.csv"
    with data_path.open(encoding="utf-8", newline="") as data_file:
        data_rows = list(csv.DictReader(data_file))

    local_times = [row["local_time_cest"] for row in data_rows]
    utc_timestamps = [row["utc_timestamp"] for row in data_rows]
    grid_load_mwh = np.array([float(row["grid_load_mwh"]) for row in data_rows])
    renewable_generation_mwh = np.array(
        [
            float(row["wind_onshore_mwh"]) + float(row["wind_offshore_mwh"]) + float(row["photovoltaics_mwh"])
            for row in data_rows
        ]
    )
    installed_capacity_mw = np.array(
        [
            float(row["wind_onshore_capacity_mw"])
            + float(row["wind_offshore_capacity_mw"])
            + float(row["photovoltaics_capacity_mw"])
            for row in data_rows
        ]
    )

    hours = np.arange(len(data_rows))
    m = hours.size
    D = grid_load_mwh / 10_000.0
    L = renewable_generation_mwh / installed_capacity_mw

    assert m == 24
    assert len(set(local_times)) == m == len(set(utc_timestamps))
    local_datetimes = [datetime.fromisoformat(timestamp) for timestamp in local_times]
    utc_datetimes = [datetime.fromisoformat(timestamp.replace("Z", "+00:00")) for timestamp in utc_timestamps]
    for timestamps in (local_datetimes, utc_datetimes):
        assert all(
            (later - earlier).total_seconds() == 3_600.0
            for earlier, later in zip(timestamps[:-1], timestamps[1:], strict=True)
        )
    assert np.all(np.isfinite(np.column_stack([grid_load_mwh, renewable_generation_mwh, installed_capacity_mw])))
    assert np.all(grid_load_mwh > 0.0)
    assert np.all(renewable_generation_mwh >= 0.0)
    assert np.all(installed_capacity_mw > 0.0)
    assert np.all((L >= 0.0) & (L <= 1.0))

    p_r = 0.08
    p_g = 0.6
    c_r = 0.4
    c_g = 2.5
    alpha_max = 12.0
    delta = 1e-3
    return (
        D,
        L,
        alpha_max,
        c_g,
        c_r,
        delta,
        grid_load_mwh,
        hours,
        installed_capacity_mw,
        m,
        p_g,
        p_r,
        renewable_generation_mwh,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the model

    `alpha` is listed as a `LowerProblem` parameter, so BLVPY treats it as
    fixed inside dispatch while retaining it as the planner's variable. Its
    planning limits are written as explicit CVXPY constraints. The initial
    `.value` provides a starting point for the nonlinear solve.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    D,
    L,
    LowerProblem,
    alpha_max,
    c_g,
    c_r,
    cp,
    delta,
    m,
    p_g,
    p_r,
):
    alpha = cp.Variable(name="alpha")
    r = cp.Variable(m, name="r")
    g = cp.Variable(m, name="g")
    alpha.value = 8.5

    dispatch = LowerProblem(
        cp.Minimize(c_r * cp.sum(r) + c_g * cp.sum(g) + delta * (cp.sum_squares(r) + cp.sum_squares(g))),
        [
            r >= 0.0,
            g >= 0.0,
            r <= cp.multiply(L, alpha),
            r + g >= D,
        ],
        parameters=[alpha],
    )
    problem = BilevelProblem(
        cp.Minimize(p_r * cp.square(alpha) + p_g * cp.sum(g)),
        dispatch,
        upper_constraints=[alpha >= 0.0, alpha <= alpha_max],
    )
    return alpha, g, problem, r


@app.cell
def _(problem):
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-5,
        conic_solver_options={
            "tol_feas": 1e-10,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
        },
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, result


@app.cell
def _(D, L, alpha, alpha_max, g, np, p_g, p_r, r, result):
    assert result.succeeded, result.message

    selected_alpha = float(np.asarray(result.variable_values[alpha]))
    selected_r = np.asarray(result.variable_values[r], dtype=float)
    selected_g = np.asarray(result.variable_values[g], dtype=float)
    closed_form_r = np.minimum(D, selected_alpha * L)
    closed_form_g = D - closed_form_r

    np.testing.assert_allclose(
        np.concatenate((selected_r, selected_g)),
        np.concatenate((closed_form_r, closed_form_g)),
        atol=1e-6,
        rtol=0.0,
    )

    alpha_grid = np.linspace(0.0, alpha_max, 2_001)
    r_grid = np.minimum(D[None, :], alpha_grid[:, None] * L[None, :])
    g_grid = D[None, :] - r_grid
    investment_costs = p_r * np.square(alpha_grid)
    thermal_penalties = p_g * np.sum(g_grid, axis=1)
    planner_objectives = investment_costs + thermal_penalties
    grid_minimum_index = int(np.argmin(planner_objectives))
    grid_minimizer = float(alpha_grid[grid_minimum_index])
    grid_spacing = float(alpha_grid[1] - alpha_grid[0])

    assert abs(selected_alpha - grid_minimizer) <= 2.0 * grid_spacing

    baseline_g = float(np.sum(D))
    optimized_g = float(np.sum(selected_g))
    return (
        alpha_grid,
        baseline_g,
        closed_form_g,
        closed_form_r,
        grid_minimizer,
        investment_costs,
        optimized_g,
        planner_objectives,
        selected_alpha,
        thermal_penalties,
    )


@app.cell(hide_code=True)
def _(
    baseline_g,
    diagnostics,
    grid_load_mwh,
    grid_minimizer,
    installed_capacity_mw,
    mo,
    optimized_g,
    result,
    selected_alpha,
):
    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | observed daily grid load | {float(grid_load_mwh.sum() / 1_000):.3f} GWh |
    | observed wind and solar capacity | {float(installed_capacity_mw[0] / 1_000):.3f} GW |
    | selected renewable capacity $10\alpha$ | {10.0 * selected_alpha:.3f} GW |
    | 2,001-point grid reference $10\alpha$ | {10.0 * grid_minimizer:.3f} GW |
    | thermal production without renewables | {10.0 * baseline_g:.3f} GWh |
    | optimized thermal production $10\mathbf{{1}}^Tg$ | {10.0 * optimized_g:.3f} GWh |
    | upper objective | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    With the observed hourly profile held fixed, the illustrative model selects
    a little more renewable capacity than the 102.823 GW installed in the
    source data. The independent grid search places its minimum at nearly the
    same capacity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpreting the capacity and dispatch

    The top panel stacks the
    modeled renewable and thermal dispatch against grid load; the dotted line
    is the renewable output physically available from the selected capacity.
    Thermal generation fills the shortfall when renewable potential is below
    demand.

    The bottom panel explains the planner's choice: investment cost rises
    quadratically, while the thermal-production penalty falls. Their sum is
    minimized close to the capacity returned by BLVPY. These grid curves use
    the hand-derived dispatch response
    $r_t(\alpha)=\min\{D_t,\alpha L_t\}$ and
    $g_t(\alpha)=D_t-r_t(\alpha)$ for this example.
    """)
    return


@app.cell(hide_code=True)
def _(
    D,
    L,
    Path,
    alpha_grid,
    closed_form_g,
    closed_form_r,
    hours,
    investment_costs,
    np,
    planner_objectives,
    plt,
    renewable_generation_mwh,
    selected_alpha,
    thermal_penalties,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, (dispatch_axis, objective_axis) = plt.subplots(2, 1, figsize=(6.5, 7))

    dispatch_axis.stackplot(
        hours,
        closed_form_r,
        closed_form_g,
        labels=[r"$r$", r"$g$"],
        colors=["C2", "grey"],
        alpha=0.85,
        step="mid",
    )
    dispatch_axis.plot(
        hours,
        D,
        color="k",
        linestyle="--",
        marker="o",
        markersize=2.5,
        drawstyle="steps-mid",
        label=r"$D$",
        zorder=10,
    )
    dispatch_axis.plot(
        hours,
        selected_alpha * L,
        color="C2",
        linestyle=":",
        marker="o",
        markersize=2.5,
        drawstyle="steps-mid",
        label=r"$\alpha L$",
    )
    dispatch_axis.plot(
        hours,
        renewable_generation_mwh / 10_000.0,
        color="C0",
        linewidth=1.0,
        marker=".",
        drawstyle="steps-mid",
        label="observed renewable output",
    )
    dispatch_axis.set(
        xlabel="Hour (CEST)",
        ylabel="Power (10 GW)",
        xticks=np.arange(0, 24, 4),
        xlim=(-0.5, 23.5),
    )
    dispatch_axis.legend(loc="upper left", frameon=False, fontsize=13)

    objective_axis.plot(alpha_grid, investment_costs, label=r"$p_r\alpha^2$", color="C2")
    objective_axis.plot(alpha_grid, thermal_penalties, label=r"$p_g\mathbf{1}^Tg$", color="grey")
    objective_axis.plot(
        alpha_grid,
        planner_objectives,
        label=r"$p_r\alpha^2+p_g\mathbf{1}^Tg$",
        color="k",
    )
    objective_axis.axvline(selected_alpha, color="C3", linestyle="--")
    objective_axis.set(
        xlabel=r"$\alpha$",
        ylabel="Planning cost",
    )
    objective_axis.legend(loc=(0, 0.1), frameon=False, fontsize=13)

    fig.tight_layout()
    figure_path = figure_dir / "renewable_capacity_planning.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
