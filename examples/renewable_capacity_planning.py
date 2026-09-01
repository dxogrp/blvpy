import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Renewable Capacity Planning with Market Dispatch

    We consider renewable capacity planning in a power system, where a
    generation planner must choose renewable capacity before an electricity
    dispatcher sees the resulting operating limits and optimizes the dispatch
    accordingly. The planner anticipates that response and trades investment
    cost against the thermal generation that remains in the dispatch.
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

    Let $\alpha\in\mathbf{R}$ denote renewable capacity, and let
    $L\in\mathbf{R}^m$ denote renewable availability over $m$ time periods.
    Thus, $\alpha L\in\mathbf{R}^m$ gives the renewable-generation limits.
    Let $D\in\mathbf{R}^m$ denote electricity demand, and let
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
    ## Representative operating data

    Twenty-four hourly periods capture a morning shoulder and an evening
    demand peak. The aggregate renewable-availability model has two distinct
    windows: a smaller morning peak and a larger afternoon peak separated by a
    midday lull.
    """)
    return


@app.cell
def _(np):
    def demand_profile(time):
        return 3.5 + 1.8 * np.exp(-0.5 * ((time - 8.0) / 2.6) ** 2) + 3.0 * np.exp(-0.5 * ((time - 19.0) / 3.2) ** 2)

    def availability_profile(time):
        morning = np.where(
            (time >= 5.0) & (time <= 11.0),
            0.8 * np.sin(np.pi * (time - 5.0) / 6.0),
            0.0,
        )
        afternoon = np.where(
            (time >= 13.0) & (time <= 21.0),
            np.sin(np.pi * (time - 13.0) / 8.0),
            0.0,
        )
        return morning + afternoon

    hours = np.arange(24)
    m = hours.size
    D = demand_profile(hours)
    L = availability_profile(hours)

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
        availability_profile,
        c_g,
        c_r,
        delta,
        demand_profile,
        m,
        p_g,
        p_r,
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
    return alpha, g, problem


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
def _(D, alpha, diagnostics, g, mo, np, result):
    assert result.succeeded, result.message

    baseline_g = float(np.sum(D))
    optimized_g = float(np.sum(g.value))
    assert optimized_g < baseline_g - 1e-3, "Renewable investment did not reduce thermal generation."

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | installed renewable capacity $\alpha$ | {float(alpha.value):.3f} |
    | thermal production without renewables | {baseline_g:.3f} |
    | optimized thermal production $\mathbf{{1}}^Tg$ | {optimized_g:.3f} |
    | upper objective | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The planner installs enough capacity to displace thermal generation during
    both renewable-availability windows, but not so much that the quadratic
    investment penalty dominates.
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

    The bottom panel explains the planner's choice: investment cost rises
    quadratically, while the thermal-production penalty falls. Their sum is
    minimized close to the capacity returned by BLVPY. These grid curves use
    the hand-derived dispatch response
    $r_t(\alpha)=\min\{D_t,\alpha L_t\}$ and
    $g_t(\alpha)=D_t-r_t(\alpha)$ for this example.
    """)
    return


@app.cell
def _(
    D,
    L,
    alpha,
    alpha_max,
    availability_profile,
    demand_profile,
    np,
    p_g,
    p_r,
    result,
):
    alpha_grid = np.linspace(0.0, alpha_max, 500)
    r_grid = np.minimum(
        D[None, :],
        alpha_grid[:, None] * L[None, :],
    )
    g_grid = D[None, :] - r_grid
    g_totals = np.sum(g_grid, axis=1)
    investment_costs = p_r * np.square(alpha_grid)
    thermal_penalties = p_g * g_totals
    planner_objectives = investment_costs + thermal_penalties
    selected_alpha = float(np.asarray(result.variable_values[alpha]))

    plot_hours = np.linspace(0.0, 23.0, 461)
    D_plot = demand_profile(plot_hours)
    L_plot = availability_profile(plot_hours)
    r_plot = np.minimum(D_plot, selected_alpha * L_plot)
    g_plot = D_plot - r_plot
    return (
        D_plot,
        L_plot,
        alpha_grid,
        g_plot,
        investment_costs,
        planner_objectives,
        plot_hours,
        r_plot,
        selected_alpha,
        thermal_penalties,
    )


@app.cell(hide_code=True)
def _(
    D_plot,
    L_plot,
    Path,
    alpha_grid,
    g_plot,
    investment_costs,
    np,
    planner_objectives,
    plot_hours,
    plt,
    r_plot,
    selected_alpha,
    thermal_penalties,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, (dispatch_axis, objective_axis) = plt.subplots(2, 1, figsize=(6.5, 7))

    dispatch_axis.stackplot(
        plot_hours,
        r_plot,
        g_plot,
        labels=[r"$r$", r"$g$"],
        colors=["C2", "grey"],
        alpha=0.85,
    )
    dispatch_axis.plot(plot_hours, D_plot, color="k", linestyle="--", label=r"$D$", zorder=10)
    dispatch_axis.plot(
        plot_hours,
        selected_alpha * L_plot,
        color="C2",
        linestyle=":",
        label=r"$\alpha L$",
    )
    dispatch_axis.set(xlabel=r"$t$", ylabel=r"Generation and demand", xticks=np.arange(0, 24, 4), xlim=(0, 23))
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
