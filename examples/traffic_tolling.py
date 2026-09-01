import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Congestion Pricing on a Braess Network

    A road authority chooses a toll before travelers select their routes
    through a network. Travelers minimize their own generalized travel cost,
    while the authority anticipates this equilibrium and minimizes total time
    spent in the network.

    The example uses a four-node Braess network. Without a toll, the apparently
    attractive central link draws too much traffic and increases congestion.
    Pricing that link can coordinate individual route choices and recover a
    nearly system-optimal allocation.
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

    Let the directed edge set be ordered as

    \[
    E=(OA,OB,AD,BD,AB),
    \]

    and let the three paths be $OAD$, $OBD$, and $OABD$. The path flow
    $x\in\mathbf{R}^3$ induces the edge flow $f\in\mathbf{R}^5$ through

    \[
    f=Hx,\qquad
    H=\begin{bmatrix}
    1&0&1\\
    0&1&0\\
    1&0&0\\
    0&1&1\\
    0&0&1
    \end{bmatrix}.
    \]

    Thus, each column of $H$ records the edges used by one path. In particular,
    only the third path uses the central edge $AB$. The fixed demand is $q$, so
    feasible path flows satisfy $x\succeq0$ and $\mathbf{1}^Tx=q$.

    Edge $e$ has affine travel time

    \[
    T_e(f_e)=b_e+a_ef_e.
    \]

    For a toll $\tau$ on $AB$, travelers reach the Wardrop equilibrium

    \[
        \begin{array}{rl}
            S(\tau) = \mathop{\rm argmin}_x & \sum_{e \in E} (b_e f_e + (1/2) a_e f_e^2) + \tau f_{AB}\\
            \mbox{subject to} & x \succeq 0,\quad \mathbf{1}^T x = q\\
            & f = Hx,
        \end{array}
    \]

    which is called the Beckmann formulation.
    Noticing that differentiating its edge term with
    respect to $f_e$ gives $T_e(f_e)$, so its optimality conditions reproduce
    travelers' route-choice conditions. The authority solves

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{\tau,x}
        & \displaystyle\sum_{e\in E}f_eT_e(f_e)+\rho\tau^2\\
    \mathop{\mathrm{subject\ to}}
        & 0\leq\tau\leq5,\\
        & x\in S(\tau).
    \end{array}
    \]

    Toll payments are transfers rather than travel time and therefore do not
    enter the first term. The small quadratic penalty selects the least costly
    toll among policies with similar network performance.
    """)
    return


@app.cell
def _(np):
    q = 6.0
    b = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    a = np.array([1.0, 0.01, 0.01, 1.0, 0.01])
    H = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rho = 0.1
    return H, a, b, q, rho


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Synthetic network

    We set $q=6$ and use

    \[
    b=(0,5,5,0,0),\qquad
    a=(1,0.01,0.01,1,0.01),\qquad \rho=0.1.
    \]

    The outer links $OA$ and $BD$ become congested quickly. The remaining
    links approximate the constant-time links of the classical Braess network.
    """)
    return


@app.cell
def _(BilevelProblem, H, LowerProblem, a, b, cp, q, rho):
    tau = cp.Variable(1, bounds=[0.0, 5.0], name="tau")
    x = cp.Variable(3, nonneg=True, name="path_flow")
    f = H @ x

    equilibrium = LowerProblem(
        cp.Minimize(0.5 * a @ cp.square(f) + b @ f + tau[0] * x[2]),
        [cp.sum(x) == q],
        parameters=[tau],
    )
    network_travel_time = b @ f + a @ cp.square(f)
    problem = BilevelProblem(
        cp.Minimize(network_travel_time + rho * cp.sum_squares(tau)),
        equilibrium,
    )

    assert problem.is_dblp()
    return problem, tau, x


@app.cell
def _(problem):
    epsilon_target = 1e-5
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=epsilon_target,
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, epsilon_target, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent reference problems

    Three ordinary convex programs provide checks on the bilevel result. The
    untolled reference fixes $\tau=0$ in the traveler problem. The fixed-toll
    reference resolves that problem at the toll selected by BLVPY. Finally, the
    system optimum minimizes $\sum_e f_eT_e(f_e)$ directly, as if a central
    controller could assign routes.

    The untolled equilibrium is not system optimal: individual travelers see
    their own travel time but not the congestion delay they impose on others.
    The central link magnifies this difference (i.e., the Braess effect).
    """)
    return


@app.cell
def _(H, a, b, cp, np, q, tau):
    def _solve_equilibrium(_tau):
        _path_flow = cp.Variable(3, nonneg=True)
        _link_flow = H @ _path_flow
        _problem = cp.Problem(
            cp.Minimize(0.5 * a @ cp.square(_link_flow) + b @ _link_flow + _tau * _path_flow[2]),
            [cp.sum(_path_flow) == q],
        )
        _problem.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=1e-11,
            tol_gap_rel=1e-11,
            tol_feas=1e-11,
        )
        assert _problem.status == cp.OPTIMAL
        return np.array(_path_flow.value, copy=True)

    untolled_path_flow = _solve_equilibrium(0.0)
    fixed_toll_path_flow = _solve_equilibrium(float(tau.value[0]))

    system_path_variable = cp.Variable(3, nonneg=True)
    system_link_expression = H @ system_path_variable
    system_problem = cp.Problem(
        cp.Minimize(b @ system_link_expression + a @ cp.square(system_link_expression)),
        [cp.sum(system_path_variable) == q],
    )
    system_problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-11,
        tol_gap_rel=1e-11,
        tol_feas=1e-11,
    )
    assert system_problem.status == cp.OPTIMAL
    system_path_flow = np.array(system_path_variable.value, copy=True)

    untolled_link_flow = H @ untolled_path_flow
    fixed_toll_link_flow = H @ fixed_toll_path_flow
    system_link_flow = H @ system_path_flow
    return (
        fixed_toll_link_flow,
        fixed_toll_path_flow,
        system_link_flow,
        system_path_flow,
        untolled_link_flow,
        untolled_path_flow,
    )


@app.cell(hide_code=True)
def _(
    H,
    a,
    b,
    diagnostics,
    epsilon_target,
    fixed_toll_link_flow,
    fixed_toll_path_flow,
    mo,
    np,
    q,
    result,
    system_link_flow,
    system_path_flow,
    tau,
    untolled_link_flow,
    untolled_path_flow,
    x,
):
    def _network_time(_link_flow):
        return float(b @ _link_flow + a @ np.square(_link_flow))

    selected_tau = float(tau.value[0])
    optimized_path_flow = np.array(x.value, copy=True)
    optimized_link_flow = H @ optimized_path_flow
    untolled_time = _network_time(untolled_link_flow)
    optimized_time = _network_time(optimized_link_flow)
    fixed_toll_time = _network_time(fixed_toll_link_flow)
    system_time = _network_time(system_link_flow)
    improvement_percent = 100.0 * (untolled_time - optimized_time) / untolled_time

    assert result.succeeded, result.message
    assert np.isfinite(result.objective)
    assert result.final_epsilon <= epsilon_target
    assert result.residuals.max_violation <= 1e-6
    assert abs(diagnostics.source_gap) <= epsilon_target
    assert -1e-8 <= selected_tau <= 5.0 + 1e-8
    assert np.min(optimized_path_flow) >= -1e-8
    assert abs(np.sum(optimized_path_flow) - q) <= 1e-7
    assert np.allclose(optimized_link_flow, H @ optimized_path_flow, atol=1e-8)
    assert np.allclose(
        fixed_toll_path_flow,
        optimized_path_flow,
        atol=1e-2,
    )
    assert np.allclose(
        fixed_toll_link_flow,
        optimized_link_flow,
        atol=1e-2,
    )
    assert abs(fixed_toll_time - optimized_time) <= 1e-2
    assert optimized_time < untolled_time - 1.0
    assert optimized_time >= system_time - 1e-5

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | selected toll $\tau^\star$ | {selected_tau:.6f} |
    | untolled path flow $x$ | {np.array2string(untolled_path_flow, precision=3)} |
    | optimized path flow $x^\star$ | {np.array2string(optimized_path_flow, precision=3)} |
    | system-optimal path flow | {np.array2string(system_path_flow, precision=3)} |
    | untolled network travel time | {untolled_time:.6f} |
    | optimized network travel time | {optimized_time:.6f} |
    | system-optimal network travel time | {system_time:.6f} |
    | travel-time improvement | {improvement_percent:.2f}% |
    | upper objective | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    With no toll, approximately $3.94$ flow units use the central path $OABD$,
    raising network travel time to {untolled_time:.2f}. The selected toll
    $\tau^\star={selected_tau:.2f}$ removes almost all flow from $AB$ and divides
    demand between $OAD$ and $OBD$. Network travel time falls by
    {improvement_percent:.1f}% to {optimized_time:.2f}, essentially matching
    the direct system optimum. The fixed-toll reference independently
    reproduces this response.
    """)
    return optimized_link_flow, optimized_path_flow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Policy and outcome comparison

    The left panels compare the untolled and optimized networks. Arrow width
    represents edge flow, and each edge is labeled with its numerical flow.
    The central edge $AB$ is highlighted because it is the tolled link.

    The right panel shows every feasible path-flow allocation

    \[
    x_1+x_2+x_3=q,\qquad x\succeq0,
    \]

    and contours report its total network travel time. The vertices assign all
    demand to one path. The markers locate the untolled equilibrium, optimized
    equilibrium, and system optimum in the complete feasible set. The latter
    two lie together at the minimum, whereas the untolled equilibrium uses the
    central path heavily and has greater total travel time.
    """)
    return


@app.cell(hide_code=True)
def _(np, optimized_link_flow, plt, untolled_link_flow):
    node_positions = {
        "O": np.array([0.0, 0.0]),
        "A": np.array([0.5, 0.42]),
        "B": np.array([0.5, -0.42]),
        "D": np.array([1.0, 0.0]),
    }
    edge_nodes = (("O", "A"), ("O", "B"), ("A", "D"), ("B", "D"), ("A", "B"))
    label_offsets = (
        np.array([-0.03, 0.08]),
        np.array([-0.03, -0.08]),
        np.array([0.03, 0.08]),
        np.array([0.03, -0.08]),
        np.array([0.08, 0.0]),
    )

    def _draw_network(_axis, _flows, _title):
        for _index, ((_tail, _head), _label_offset) in enumerate(zip(edge_nodes, label_offsets)):
            _start = node_positions[_tail]
            _end = node_positions[_head]
            _is_central = _index == 4
            _axis.annotate(
                "",
                xy=_end,
                xytext=_start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "C3" if _is_central else "0.35",
                    "linewidth": 0.9 + 0.55 * _flows[_index],
                    "mutation_scale": 10,
                    "shrinkA": 9,
                    "shrinkB": 9,
                },
                zorder=2,
            )
            _midpoint = 0.5 * (_start + _end) + _label_offset
            _axis.text(
                *_midpoint,
                rf"${_flows[_index]:.2f}$",
                ha="center",
                va="center",
                fontsize=10,
            )

        for _node, _position in node_positions.items():
            _axis.scatter(*_position, s=55, color="black", zorder=4)
            _axis.text(
                _position[0],
                _position[1] + (0.09 if _node in {"A", "D"} else -0.09),
                rf"${_node}$",
                ha="center",
                va="center",
                fontsize=10,
            )

        _axis.set(
            xlim=(-0.08, 1.08),
            ylim=(-0.62, 0.62),
            aspect="equal",
        )
        _axis.set_title(_title, fontsize=15, pad=0)
        _axis.axis("off")

    def make_network_figure():
        _figure = plt.figure(figsize=(7.5, 5), layout="constrained")
        _grid = _figure.add_gridspec(2, 2, width_ratios=[1, 1.5])
        _untolled_axis = _figure.add_subplot(_grid[0, 0])
        _optimized_axis = _figure.add_subplot(_grid[1, 0])
        _outcome_axis = _figure.add_subplot(_grid[:, 1])

        _draw_network(_untolled_axis, untolled_link_flow, r"$\tau=0$")
        _draw_network(
            _optimized_axis,
            optimized_link_flow,
            r"$\tau=\tau^\star$",
        )
        return _figure, _outcome_axis

    return (make_network_figure,)


@app.cell(hide_code=True)
def _(
    H,
    Path,
    a,
    b,
    make_network_figure,
    np,
    optimized_path_flow,
    plt,
    q,
    system_path_flow,
    untolled_path_flow,
):
    simplex_figure, simplex_axis = make_network_figure()

    simplex_resolution = 40
    simplex_path_flow = []
    for _index_1 in range(simplex_resolution + 1):
        for _index_2 in range(simplex_resolution + 1 - _index_1):
            _x_1 = q * _index_1 / simplex_resolution
            _x_2 = q * _index_2 / simplex_resolution
            _x_3 = q - _x_1 - _x_2
            simplex_path_flow.append((_x_1, _x_2, _x_3))
    simplex_path_flow = np.asarray(simplex_path_flow)

    simplex_vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
        ]
    )
    simplex_coordinates = (simplex_path_flow / q) @ simplex_vertices
    simplex_link_flow = simplex_path_flow @ H.T
    simplex_network_time = simplex_link_flow @ b + np.sum(
        a * np.square(simplex_link_flow),
        axis=1,
    )

    filled_contours = simplex_axis.tricontourf(
        simplex_coordinates[:, 0],
        simplex_coordinates[:, 1],
        simplex_network_time,
        levels=10,
        cmap="Greys",
        alpha=0.75,
    )
    contour_lines = simplex_axis.tricontour(
        simplex_coordinates[:, 0],
        simplex_coordinates[:, 1],
        simplex_network_time,
        levels=filled_contours.levels,
        colors="black",
        linewidths=0.45,
        alpha=0.55,
    )
    simplex_axis.clabel(contour_lines, inline=True, fontsize=7, fmt="%.0f")

    closed_vertices = np.vstack([simplex_vertices, simplex_vertices[0]])
    simplex_axis.plot(
        closed_vertices[:, 0],
        closed_vertices[:, 1],
        color="black",
        linewidth=1.0,
    )

    def _simplex_position(_path_flow):
        return (_path_flow / q) @ simplex_vertices

    untolled_position = _simplex_position(untolled_path_flow)
    optimized_position = _simplex_position(optimized_path_flow)
    system_position = _simplex_position(system_path_flow)

    simplex_axis.scatter(
        *untolled_position,
        color="C0",
        edgecolor="white",
        linewidth=0.6,
        marker="o",
        s=55,
        zorder=5,
        label=r"$x^{\tau=0}$",
    )
    simplex_axis.scatter(
        *system_position,
        facecolors="none",
        edgecolors="C2",
        marker="*",
        linewidths=1.8,
        s=210,
        zorder=6,
        label=r"$x^{\rm sys}$",
    )
    simplex_axis.scatter(
        *optimized_position,
        color="C3",
        marker="X",
        s=70,
        zorder=7,
        label=r"$x^{\tau=\tau^\star}$",
    )

    simplex_axis.text(
        0,
        -0.045,
        r"$x_1=q$",
        ha="right",
        va="top",
        fontsize=12,
    )
    simplex_axis.text(
        0,
        -0.115,
        r"$OAD$",
        ha="right",
        va="top",
        fontsize=12,
    )
    simplex_axis.text(
        1,
        -0.045,
        r"$x_2=q$",
        ha="left",
        va="top",
        fontsize=12,
    )
    simplex_axis.text(
        1,
        -0.115,
        r"$OBD$",
        ha="left",
        va="top",
        fontsize=12,
    )
    simplex_axis.text(
        0.5,
        simplex_vertices[2, 1] + 0.13,
        r"$x_3=q$",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    simplex_axis.text(
        0.5,
        simplex_vertices[2, 1] + 0.055,
        r"$OABD$",
        ha="center",
        va="bottom",
        fontsize=12,
    )
    simplex_axis.set(
        xlim=(-0.15, 1.15),
        ylim=(-0.18, 1.05),
        aspect="equal",
    )
    simplex_axis.axis("off")
    simplex_axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=15,
    )
    simplex_colorbar = simplex_figure.colorbar(
        filled_contours,
        ax=simplex_axis,
        fraction=0.04,
        pad=0.03,
    )
    simplex_colorbar.set_label(r"$\sum_{e\in E} f_eT_e(f_e)$", fontsize=12)
    simplex_colorbar.ax.tick_params(labelsize=12)

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "traffic_tolling.pdf"
    simplex_figure.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
