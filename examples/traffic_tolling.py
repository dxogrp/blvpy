import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Congestion Pricing on Parallel Routes

    Consider two roads connecting the same origin and destination. A fixed
    amount of traffic must make the trip, and every traveler chooses one of the
    roads. The road authority cannot assign routes directly, but it can influence
    those choices by charging tolls.

    This creates two linked optimization problems. In the **upper problem**, the
    authority chooses tolls to reduce congestion. In the **lower problem**,
    travelers observe those tolls and choose routes that minimize their own
    perceived cost.
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
    ## Travel time and individual route choice

    Let $D$ be the total travel demand and $f_i$ the aggregate flow using route
    $i$. The flows are continuous here: one unit can represent one vehicle, one
    thousand vehicles, or any other consistent traffic unit. Because every trip
    uses exactly one route,

    \[
    f_i\geq0,\qquad \sum_i f_i=D.
    \]

    Travelers are treated as **nonatomic**: each individual is too small to
    change congestion alone and therefore takes the current route times as
    given when choosing a road.

    Congestion makes a route slower as more travelers use it. We use the affine
    travel-time function

    \[
    T_i(f_i)=b_i+a_i f_i,
    \]

    where $b_i$ is the free-flow time when the road is empty and $a_i$ measures
    how quickly congestion grows. A traveler perceives the generalized cost
    $T_i(f_i)+\tau_i$, where the toll $\tau_i$ is expressed in time-equivalent
    units.

    A **Wardrop equilibrium** is reached when no traveler can improve their
    perceived cost by switching routes. Thus every used route has the same
    minimum generalized cost; an unused route cannot have a lower cost.

    ## The traveler's lower problem

    For separable increasing travel times, the Wardrop conditions are exactly
    the optimality conditions of the Beckmann problem

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_f
        & \displaystyle\sum_i\left(b_i f_i+\frac12a_i f_i^2+\tau_i f_i\right)\\
    \mathop{\mathrm{subject\ to}}
        & f\geq0,\quad \mathbf{1}^Tf=D.
    \end{array}
    \]

    Notice that the objective derivative with respect to one route's flow is

    \[
    \frac{d}{df_i}\left(b_if_i+\frac12a_if_i^2+\tau_if_i\right)
      =T_i(f_i)+\tau_i.
    \]

    Consequently, its optimality conditions equalize travelers' perceived
    route costs. `LowerProblem(parameters=[toll])` tells BLVPY to hold the tolls
    fixed while solving this equilibrium problem.

    ## The road authority's upper problem

    The authority cares about actual time spent by all travelers, i.e.,
    $\sum_i f_iT_i(f_i)$. It anticipates the equilibrium response and solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{\tau,f}
        & \displaystyle\sum_i f_iT_i(f_i)+0.01\lVert\tau\rVert_2^2\\
    \mathop{\mathrm{subject\ to}}
        & 0\leq\tau_i\leq5,\\
        & f\text{ is a Wardrop equilibrium under }\tau.
    \end{array}
    \]

    Toll payments are transfers rather than travel time, so they are not added
    directly to the performance measure. The small quadratic term only
    discourages tolls larger than needed to induce a useful route allocation.
    """)
    return


@app.cell
def _(np):
    demand = 10.0
    free_flow_time = np.array([1.0, 2.0])
    congestion_slope = np.array([0.20, 0.05])
    route_names = ("route 1", "route 2")
    return congestion_slope, demand, free_flow_time, route_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical scenario and BLVPY model

    Ten units of traffic must be allocated. Route 1 has free-flow time $1$ but
    congestion slope $0.20$; route 2 starts slower, at time $2$, but has the
    gentler slope $0.05$. Route 1 is therefore attractive at low flow but becomes
    congested much faster.

    Without tolls, travelers overuse the initially faster route because each
    traveler considers their own trip time but not the delay their presence adds
    for everyone else. A toll on route 1 can represent that congestion
    externality and move some flow toward route 2. In the code, `toll` is the
    upper variable and `flow` contains the two lower equilibrium flows.
    """)
    return


@app.cell
def _(
    BilevelProblem,
    LowerProblem,
    congestion_slope,
    cp,
    demand,
    free_flow_time,
):
    toll = cp.Variable(2, bounds=[0.0, 5.0], name="toll")
    flow = cp.Variable(2, nonneg=True, name="route_flow")

    equilibrium = LowerProblem(
        cp.Minimize(0.5 * cp.sum(cp.multiply(congestion_slope, cp.square(flow))) + (free_flow_time + toll) @ flow),
        [cp.sum(flow) == demand],
        parameters=[toll],
    )
    total_travel_time = free_flow_time @ flow + cp.sum(cp.multiply(congestion_slope, cp.square(flow)))
    problem = BilevelProblem(
        cp.Minimize(total_travel_time + 0.01 * cp.sum_squares(toll)),
        equilibrium,
    )
    return flow, problem, toll


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
def _(mo):
    mo.md(r"""
    ## Reference scenarios

    We compute two benchmarks with ordinary convex optimization:

    1. The **untolled equilibrium** sets $\tau=0$ and lets travelers choose. Both
       routes are used, so their travel times become equal. Solving
       $T_1(f_1)=T_2(f_2)$ with $f_1+f_2=10$ gives $(f_1,f_2)=(6,4)$.
    2. The **system optimum** imagines a central controller that can assign every
       traveler directly. It minimizes total travel time
       $\sum_i f_iT_i(f_i)$ and produces $(f_1,f_2)=(4,6)$.

    Travelers respond to their own route time
    $T_i(f_i)=b_i+a_if_i$. A central planner instead considers the marginal
    increase in the time spent by everyone on that route. Because total route
    time is

    \[
    f_iT_i(f_i)=b_if_i+a_if_i^2,
    \]

    its derivative with respect to flow is

    \[
    \frac{d}{df_i}\left[f_iT_i(f_i)\right]
      =T_i(f_i)+f_iT_i'(f_i)
      =b_i+2a_if_i.
    \]

    The additional term $a_if_i$ is the congestion delay that new traffic
    imposes on travelers already using the route. Individual travelers do not
    account for this external delay, while the system planner does. The system
    optimum is therefore a benchmark for network performance.
    """)
    return


@app.cell
def _(congestion_slope, cp, demand, free_flow_time, np):
    untolled_variable = cp.Variable(2, nonneg=True)
    untolled_problem = cp.Problem(
        cp.Minimize(
            0.5 * cp.sum(cp.multiply(congestion_slope, cp.square(untolled_variable)))
            + free_flow_time @ untolled_variable
        ),
        [cp.sum(untolled_variable) == demand],
    )
    untolled_problem.solve(solver=cp.CLARABEL)
    untolled_flow = np.array(untolled_variable.value, copy=True)

    system_variable = cp.Variable(2, nonneg=True)
    system_problem = cp.Problem(
        cp.Minimize(
            free_flow_time @ system_variable + cp.sum(cp.multiply(congestion_slope, cp.square(system_variable)))
        ),
        [cp.sum(system_variable) == demand],
    )
    system_problem.solve(solver=cp.CLARABEL)
    system_flow = np.array(system_variable.value, copy=True)
    return system_flow, untolled_flow


@app.cell(hide_code=True)
def _(
    congestion_slope,
    diagnostics,
    flow,
    free_flow_time,
    mo,
    np,
    result,
    system_flow,
    toll,
    untolled_flow,
):
    def _travel_time(_flow):
        return float(free_flow_time @ _flow + congestion_slope @ np.square(_flow))

    optimized_flow = np.array(flow.value, copy=True)
    optimized_route_times = free_flow_time + congestion_slope * optimized_flow
    optimized_perceived_costs = optimized_route_times + toll.value
    untolled_time = _travel_time(untolled_flow)
    optimized_time = _travel_time(optimized_flow)
    system_time = _travel_time(system_flow)
    improvement_percent = 100.0 * (untolled_time - optimized_time) / untolled_time

    assert result.succeeded, result.message
    assert optimized_time < untolled_time - 1e-3, "Optimized tolls did not improve on the untolled equilibrium."
    assert optimized_time <= system_time + 1e-3, (
        "The optimized equilibrium is inconsistent with the direct system optimum."
    )

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | untolled route flows | {np.array2string(untolled_flow, precision=3)} |
    | system-optimal route flows | {np.array2string(system_flow, precision=3)} |
    | optimized tolls | {np.array2string(toll.value, precision=3)} |
    | optimized route flows | {np.array2string(optimized_flow, precision=3)} |
    | optimized route travel times | {np.array2string(optimized_route_times, precision=3)} |
    | optimized perceived costs | {np.array2string(optimized_perceived_costs, precision=3)} |
    | untolled total travel time | {untolled_time:.6f} |
    | optimized total travel time | {optimized_time:.6f} |
    | system-optimal travel time | {system_time:.6f} |
    | travel-time improvement | {improvement_percent:.2f}% |
    | upper objective | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    At the untolled equilibrium, six flow units choose route 1 and four choose
    route 2; both routes then take time $2.2$. The optimized toll of about $0.5$
    on route 1 shifts roughly two units toward route 2. Physical travel times no
    longer match, but travel time plus toll is almost equal across the two used
    routes, as Wardrop equilibrium requires. The resulting allocation nearly
    reproduces the system optimum and lowers aggregate travel time from $22$ to
    $21$, an improvement of about $4.5\%$.

    The toll is a coordination signal, not proof that every traveler is better
    off individually. Distributional effects and toll revenues are outside this
    educational model. The BLVPY result is also a local numerical solution, not
    a global optimality certificate.
    """)
    return optimized_flow, optimized_time, system_time, untolled_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Equilibrium comparison

    The two network diagrams use line thickness to represent route flow. Without
    tolls, more traffic chooses the initially faster first route. Under the
    optimized toll, approximately two flow units move to the less
    congestion-sensitive second route. Each route is annotated with its flow,
    physical travel time, and toll.

    The equilibrium-cost panel uses route-1 flow $f_1$ on its horizontal axis;
    route-2 flow is then $D-f_1$. With no toll, the two physical travel-time
    curves intersect at $f_1=6$. Adding the route-1 toll shifts its perceived
    cost upward, moving the Wardrop intersection to approximately $f_1=4$.

    Finally, the network-time curve evaluates total time spent by all travelers
    for every possible split. Its minimum is the system optimum. The optimized
    equilibrium lies almost exactly at this minimum, while the untolled
    equilibrium lies higher on the curve. This shows both how the toll changes
    voluntary route choice and why that change improves network performance.
    """)
    return


@app.cell(hide_code=True)
def _(
    Path,
    congestion_slope,
    demand,
    free_flow_time,
    np,
    optimized_flow,
    optimized_time,
    plt,
    route_names,
    system_flow,
    system_time,
    toll,
    untolled_flow,
    untolled_time,
):
    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    route_colors = ("#377eb8", "#ff7f00")

    def _draw_network(axis, flows, tolls, title):
        path_position = np.linspace(0.0, 1.0, 200)
        route_paths = (
            0.27 * np.sin(np.pi * path_position),
            -0.27 * np.sin(np.pi * path_position),
        )
        route_times = free_flow_time + congestion_slope * flows
        for index, (path, color) in enumerate(zip(route_paths, route_colors)):
            axis.plot(
                path_position,
                path,
                color=color,
                linewidth=1.8 + 0.75 * flows[index],
                solid_capstyle="round",
                alpha=0.85,
            )
            label_height = 0.39 if index == 0 else -0.39
            axis.text(
                0.5,
                label_height,
                (
                    f"{route_names[index]}: flow={flows[index]:.2f}, "
                    f"time={route_times[index]:.2f}, toll={tolls[index]:.2f}"
                ),
                ha="center",
                va="center",
                fontsize=8,
            )
        axis.scatter([0.0, 1.0], [0.0, 0.0], s=90, color="#333333", zorder=5)
        axis.text(0.0, -0.1, "origin", ha="center", va="top", fontsize=8)
        axis.text(1.0, -0.1, "destination", ha="center", va="top", fontsize=8)
        axis.set(xlim=(-0.12, 1.12), ylim=(-0.52, 0.52), title=title)
        axis.axis("off")

    fig = plt.figure(figsize=(10.0, 7.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15], hspace=0.28, wspace=0.25)
    untolled_axis = fig.add_subplot(grid[0, 0])
    optimized_axis = fig.add_subplot(grid[0, 1])
    equilibrium_axis = fig.add_subplot(grid[1, 0])
    network_time_axis = fig.add_subplot(grid[1, 1])

    _draw_network(untolled_axis, untolled_flow, np.zeros(2), "Untolled equilibrium")
    _draw_network(optimized_axis, optimized_flow, toll.value, "Optimized-toll equilibrium")

    route_1_flow = np.linspace(0.0, demand, 401)
    route_2_flow = demand - route_1_flow
    route_1_time = free_flow_time[0] + congestion_slope[0] * route_1_flow
    route_2_time = free_flow_time[1] + congestion_slope[1] * route_2_flow
    route_1_tolled_cost = route_1_time + toll.value[0]

    equilibrium_axis.plot(route_1_flow, route_1_time, color=route_colors[0], label=r"$T_1(f_1)$")
    equilibrium_axis.plot(
        route_1_flow,
        route_2_time,
        color=route_colors[1],
        label=r"$T_2(D-f_1)$",
    )
    equilibrium_axis.plot(
        route_1_flow,
        route_1_tolled_cost,
        color=route_colors[0],
        linestyle="--",
        label=r"$T_1(f_1)+\tau_1$",
    )
    equilibrium_axis.scatter(
        untolled_flow[0],
        free_flow_time[0] + congestion_slope[0] * untolled_flow[0],
        color="#333333",
        marker="o",
        s=55,
        zorder=5,
        label="untolled equilibrium",
    )
    equilibrium_axis.scatter(
        optimized_flow[0],
        free_flow_time[0] + congestion_slope[0] * optimized_flow[0] + toll.value[0],
        color="#e41a1c",
        marker="X",
        s=75,
        zorder=6,
        label="optimized equilibrium",
    )
    equilibrium_axis.set(
        xlabel=r"route-1 flow $f_1$",
        ylabel="travel time plus toll",
        title="Generalized Costs and Wardrop Equilibria",
        xlim=(0.0, demand),
    )
    equilibrium_axis.legend(frameon=False, fontsize=8)

    total_time_curve = route_1_flow * route_1_time + route_2_flow * route_2_time
    network_time_axis.plot(route_1_flow, total_time_curve, color="#4daf4a", linewidth=2.2)
    network_time_axis.scatter(
        untolled_flow[0],
        untolled_time,
        color="#333333",
        marker="o",
        s=65,
        zorder=5,
        label="untolled equilibrium",
    )
    network_time_axis.scatter(
        system_flow[0],
        system_time,
        facecolors="none",
        edgecolors="#377eb8",
        marker="*",
        linewidths=1.8,
        s=210,
        zorder=6,
        label="system optimum",
    )
    network_time_axis.scatter(
        optimized_flow[0],
        optimized_time,
        color="#e41a1c",
        marker="X",
        s=75,
        zorder=7,
        label="optimized equilibrium",
    )
    network_time_axis.set(
        xlabel=r"route-1 flow $f_1$",
        ylabel="total network travel time",
        title="Network Travel Time by Flow Allocation",
        xlim=(0.0, demand),
    )
    network_time_axis.legend(frameon=False, fontsize=8)

    figure_path = figure_dir / "traffic_tolling.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
