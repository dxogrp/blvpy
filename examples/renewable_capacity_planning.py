import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Renewable Capacity Planning with Market Dispatch

    A generation planner must choose renewable capacity before an electricity
    dispatcher sees the resulting operating limits. The dispatcher is the
    lower-level decision maker: for the installed capacity, it minimizes
    operating cost while meeting demand in every representative time block.
    The planner anticipates that response and trades investment cost against
    the thermal generation that remains in the dispatch.
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

    Let $k$ denote installed renewable capacity, $a_t$ its availability,
    and $d_t$ demand. For a fixed $k$, the dispatcher chooses renewable
    output $r_t$ and thermal output $g_t$:

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{r,g}
        & c_r\mathbf{1}^Tr+c_g\mathbf{1}^Tg
          +\delta(\|r\|_2^2+\|g\|_2^2) \\
    \mathop{\mathrm{subject\ to}}
        & 0\leq r_t\leq a_t k,\quad g_t\geq0,\\
        & r_t+g_t\geq d_t.
    \end{array}
    \]

    The small quadratic term makes the lower response stable. Anticipating this
    dispatch, the planner solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{k,r,g}
        & 0.08k^2+0.6\mathbf{1}^Tg \\
    \mathop{\mathrm{subject\ to}}
        & 0\leq k\leq12,\\
        & (r,g)\text{ solves the dispatch problem for }k.
    \end{array}
    \]

    The two upper-level terms represent increasing marginal investment cost
    and a penalty on thermal production.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Representative operating data

    Twelve two-hour blocks capture a morning shoulder, an evening peak, and a
    daylight-only renewable profile.
    """)
    return


@app.cell
def _(np):
    hours = np.arange(0, 24, 2)
    block = np.arange(hours.size)
    demand = 3.5 + 1.8 * np.exp(-0.5 * ((block - 4.0) / 1.3) ** 2) + 3.0 * np.exp(-0.5 * ((block - 9.5) / 1.6) ** 2)
    availability = np.maximum(np.sin(np.pi * (block - 3.0) / 6.0), 0.0)
    return availability, demand, hours


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the model

    `capacity` is listed as a `LowerProblem` parameter, so BLVPY treats it as
    fixed inside dispatch while retaining it as the planner's variable. Its
    planning limits are written as explicit CVXPY constraints. The initial
    `.value` provides a starting point for the nonlinear solve.
    """)
    return


@app.cell
def _(BilevelProblem, LowerProblem, availability, cp, demand):
    capacity = cp.Variable(name="renewable_capacity")
    renewable = cp.Variable(demand.size, name="renewable_dispatch")
    thermal = cp.Variable(demand.size, name="thermal_dispatch")
    capacity.value = 6.0

    dispatch = LowerProblem(
        cp.Minimize(
            0.4 * cp.sum(renewable)
            + 2.5 * cp.sum(thermal)
            + 1e-3 * (cp.sum_squares(renewable) + cp.sum_squares(thermal))
        ),
        [
            renewable >= 0.0,
            thermal >= 0.0,
            renewable <= cp.multiply(availability, capacity),
            renewable + thermal >= demand,
        ],
        parameters=[capacity],
    )
    problem = BilevelProblem(
        cp.Minimize(0.08 * cp.square(capacity) + 0.6 * cp.sum(thermal)),
        dispatch,
        upper_constraints=[capacity >= 0.0, capacity <= 12.0],
    )
    return capacity, problem, renewable, thermal


@app.cell
def _(problem):
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=1e-5,
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, result


@app.cell(hide_code=True)
def _(capacity, demand, diagnostics, mo, np, result, thermal):
    assert result.succeeded, result.message

    baseline_thermal = float(np.sum(demand))
    optimized_thermal = float(np.sum(thermal.value))
    assert optimized_thermal < baseline_thermal - 1e-3, "Renewable investment did not reduce thermal generation."

    mo.md(f"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | installed renewable capacity | {float(capacity.value):.3f} |
    | thermal production without renewables | {baseline_thermal:.3f} |
    | optimized thermal production | {optimized_thermal:.3f} |
    | upper objective | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The planner installs enough capacity to displace thermal generation during
    daylight, but not so much that the quadratic investment penalty dominates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpreting the capacity and dispatch

    The top panel stacks renewable and thermal output against demand. The
    dotted line is the renewable output physically available from the selected
    capacity; thermal generation fills the shortfall when renewable potential
    is below demand.

    The lower-left panel shows how total thermal production falls as capacity
    is added. The curve flattens because renewable availability is low or zero
    in some blocks, so capacity alone cannot displace all thermal generation.

    The lower-right panel explains the planner's choice: investment cost rises
    quadratically, while the thermal-production penalty falls. Their sum is
    minimized close to the capacity returned by BLVPY. These grid curves use
    the hand-derived dispatch response
    $r_t(k)=\min\{d_t,a_tk\}$ and $g_t(k)=d_t-r_t(k)$ for this example.
    """)
    return


@app.cell
def _(availability, capacity, demand, np, result):
    capacity_grid = np.linspace(0.0, 12.0, 241)
    renewable_grid = np.minimum(
        demand[None, :],
        capacity_grid[:, None] * availability[None, :],
    )
    thermal_grid = demand[None, :] - renewable_grid
    thermal_totals = np.sum(thermal_grid, axis=1)
    investment_costs = 0.08 * np.square(capacity_grid)
    thermal_penalties = 0.6 * thermal_totals
    planner_objectives = investment_costs + thermal_penalties
    selected_capacity = float(np.asarray(result.variable_values[capacity]))
    return (
        capacity_grid,
        investment_costs,
        planner_objectives,
        selected_capacity,
        thermal_penalties,
        thermal_totals,
    )


@app.cell
def _(
    Path,
    availability,
    capacity_grid,
    demand,
    hours,
    investment_costs,
    np,
    planner_objectives,
    plt,
    renewable,
    selected_capacity,
    thermal,
    thermal_penalties,
    thermal_totals,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.0, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    dispatch_axis = fig.add_subplot(grid[0, :])
    thermal_axis = fig.add_subplot(grid[1, 0])
    objective_axis = fig.add_subplot(grid[1, 1])

    dispatch_axis.stackplot(
        hours,
        renewable.value,
        thermal.value,
        labels=["renewable dispatch", "thermal dispatch"],
        colors=["#4daf4a", "#777777"],
        alpha=0.85,
    )
    dispatch_axis.plot(hours, demand, color="black", linestyle="--", linewidth=1.8, label="demand")
    dispatch_axis.plot(
        hours,
        selected_capacity * availability,
        color="#377eb8",
        linestyle=":",
        linewidth=1.8,
        label="available renewable output",
    )
    dispatch_axis.set(
        xlabel="hour of day",
        ylabel="power per two-hour block",
        title=f"Dispatch at selected renewable capacity {selected_capacity:.2f}",
        xticks=np.arange(0, 24, 4),
    )
    dispatch_axis.legend(loc="upper left", frameon=False, ncols=2)

    thermal_axis.plot(capacity_grid, thermal_totals, color="#777777", linewidth=2.0)
    thermal_axis.axvline(selected_capacity, color="#e41a1c", linestyle="--", label="BLVPY capacity")
    thermal_axis.set(
        xlabel="renewable capacity",
        ylabel="total thermal production",
        title="Thermal displacement",
    )
    thermal_axis.legend(frameon=False)

    objective_axis.plot(capacity_grid, investment_costs, label="investment cost", color="#377eb8")
    objective_axis.plot(capacity_grid, thermal_penalties, label="thermal penalty", color="#777777")
    objective_axis.plot(
        capacity_grid, planner_objectives, label="total upper objective", color="#4daf4a", linewidth=2.0
    )
    objective_axis.axvline(selected_capacity, color="#e41a1c", linestyle="--", label="BLVPY capacity")
    objective_axis.set(
        xlabel="renewable capacity",
        ylabel="objective contribution",
        title="Planner's tradeoff",
    )
    objective_axis.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    figure_path = figure_dir / "renewable_capacity_planning.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
