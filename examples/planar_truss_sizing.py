import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Planar Truss Sizing under Elastic Equilibrium

    A structural designer allocates material among the members of a planar
    truss before the structure deforms under load. The design must anticipate
    elastic equilibrium: changing a member area changes the stiffness, which
    redistributes both displacement and internal force throughout the truss.

    This example sizes a two-bay cantilever. The upper problem minimizes
    compliance subject to a material budget, while the lower problem finds the
    displacement that minimizes total potential energy.
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

    Suppose the truss has $n$ members and $m$ nodal displacement degrees of
    freedom. Let $a,L\in\mathbf{R}^n$ collect the member areas and lengths, and
    let the scalar $E>0$ be the Young's modulus shared by all members. The
    vectors $u,f\in\mathbf{R}^m$ collect nodal displacements and applied nodal
    forces. The compatibility matrix $B\in\mathbf{R}^{n\times m}$ maps nodal
    displacements to axial member elongations, so its $e$th component is
    $(Bu)_e=b_e^Tu$.

    The designer chooses the areas while anticipating elastic equilibrium:

    \[
    \begin{array}{ll}
    \mathop{\mathrm{minimize}}_{a,u} & f^Tu\\
    \mathop{\mathrm{subject\ to}}
        & a_{\min}\mathbf{1}\preceq a\preceq a_{\max}\mathbf{1},\\
        & L^Ta\leq V_{\max},\\
        & u\in S(a).
    \end{array}
    \]

    Here $0<a_{\min}\leq a_{\max}$ bound each cross-sectional area, and
    $V_{\max}>0$ is the material-volume budget because member $e$ has volume
    $L_ea_e$. For the prescribed load $f$, the objective $f^Tu$ is compliance:
    smaller compliance means a stiffer structure.

    For an area allocation $a$, the assembled stiffness matrix is

    \[
    K(a)=B^T\mathop{\rm diag}\left(
        \frac{Ea_1}{L_1},\ldots,\frac{Ea_n}{L_n}
    \right)B.
    \]

    Let $\mathcal F$ contain the fixed displacement indices. The lower-level elasticity problem is

    \[
    \begin{array}{ll}
    S(a)=\mathop{\mathrm{argmin}}_u
        & \displaystyle\frac{1}{2}u^TK(a)u-f^Tu\\
    \mathop{\mathrm{subject\ to}}
        & u_i=0,\qquad i\in\mathcal F.
    \end{array}
    \]

    This is the minimum-total-potential-energy principle for a pin-jointed,
    small-displacement, linear-elastic truss. The strain-energy term also has
    the equivalent memberwise form

    \[
    \frac{1}{2}u^TK(a)u
    =\frac{1}{2}\sum_{e=1}^n\frac{Ea_e}{L_e}(b_e^Tu)^2.
    \]

    Partitioning the displacement into free and fixed components gives

    \[
    K_{\rm free,free}(a)u_{\rm free}=f_{\rm free},
    \qquad u_{\rm fixed}=0.
    \]

    Thus the optimization enforces ordinary static equilibrium.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two-bay cantilever

    Six nodes form two square bays with crossed diagonals. Both degrees of
    freedom at the two left nodes, numbered 0 and 1, are fixed. With
    $u=(u_{0x},u_{0y},u_{1x},u_{1y},\ldots)$, this means
    $\mathcal F=\{0,1,2,3\}$. A unit downward load acts at the lower tip and a
    smaller downward load acts at the upper tip. All quantities are normalized:
    $E=10$, $0.08\leq a_e\leq2$, and the material budget is $65\%$ of the
    volume obtained by assigning unit area to every member.
    """)
    return


@app.cell
def _(np):
    nodes = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    members = np.array(
        [
            [0, 2],
            [1, 3],
            [2, 3],
            [0, 3],
            [1, 2],
            [2, 4],
            [3, 5],
            [4, 5],
            [2, 5],
            [3, 4],
        ]
    )

    n_nodes = nodes.shape[0]
    n = members.shape[0]
    m = 2 * n_nodes
    B = np.zeros((n, m))
    L = np.zeros(n)
    for member_index, (start_node, end_node) in enumerate(members):
        direction = nodes[end_node] - nodes[start_node]
        L[member_index] = np.linalg.norm(direction)
        direction /= L[member_index]
        B[member_index, 2 * start_node : 2 * start_node + 2] = -direction
        B[member_index, 2 * end_node : 2 * end_node + 2] = direction

    f = np.zeros(m)
    f[2 * 4 + 1] = -1.0
    f[2 * 5 + 1] = -0.25
    fixed_nodes = np.array([0, 1])
    F = np.column_stack((2 * fixed_nodes, 2 * fixed_nodes + 1)).ravel()

    E = 10.0
    a_min = 0.08
    a_max = 2.0
    V_max = 0.65 * np.sum(L)
    return (
        B,
        E,
        F,
        L,
        V_max,
        a_max,
        a_min,
        f,
        fixed_nodes,
        m,
        members,
        n,
        n_nodes,
        nodes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and solve the model

    `a` is an upper-level variable and is listed as a lower-problem
    parameter, which is therefore fixed when equilibrium is computed.
    A uniform feasible design supplies the nonlinear solver's
    initial point.
    """)
    return


@app.cell
def _(
    B,
    BilevelProblem,
    E,
    F,
    L,
    LowerProblem,
    V_max,
    a_max,
    a_min,
    cp,
    f,
    m,
    n,
    np,
):
    a = cp.Variable(
        n,
        nonneg=True,
        bounds=[a_min, a_max],
        name="a",
    )
    u = cp.Variable(m, name="u")
    a.value = np.full(n, V_max / np.sum(L))

    stiffness_weights = cp.multiply(E / L, a)
    strain_energy = 0.5 * cp.sum(cp.multiply(stiffness_weights, cp.square(B @ u)))
    equilibrium = LowerProblem(
        cp.Minimize(strain_energy - f @ u),
        [u[F] == 0.0],
        parameters=[a],
    )
    problem = BilevelProblem(
        cp.Minimize(f @ u),
        equilibrium,
        upper_constraints=[L @ a <= V_max],
    )

    assert problem.is_dblp()
    return a, problem, u


@app.cell
def _(problem):
    epsilon_target = 1e-9
    result = problem.solve(
        epsilon_initial=1e-2,
        epsilon_target=epsilon_target,
        best_of=2,
        seed=7,
        verbose=False,
    )
    diagnostics = problem.gap_diagnostics(result)
    return diagnostics, epsilon_target, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Independent elastic-equilibrium references

    Two ordinary CVXPY problems independently resolve equilibrium at fixed
    areas. The first uses the uniform allocation, providing a baseline with
    the same material volume. The second fixes the areas selected by BLVPY and
    verifies its lower-level displacement without using the lifted KKT model.
    """)
    return


@app.cell
def _(B, E, F, L, V_max, a, cp, f, m, n, np):
    def solve_fixed_equilibrium(a_value):
        reference_u = cp.Variable(m)
        reference_weights = E * a_value / L
        reference_energy = 0.5 * cp.sum(cp.multiply(reference_weights, cp.square(B @ reference_u)))
        reference_problem = cp.Problem(
            cp.Minimize(reference_energy - f @ reference_u),
            [reference_u[F] == 0.0],
        )
        reference_problem.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=1e-11,
            tol_gap_rel=1e-11,
            tol_feas=1e-11,
        )
        assert reference_problem.status == cp.OPTIMAL
        return np.asarray(reference_u.value, dtype=float)

    uniform_a = np.full(n, V_max / np.sum(L))
    uniform_u = solve_fixed_equilibrium(uniform_a)
    optimized_a = np.asarray(a.value, dtype=float)
    optimized_reference_u = solve_fixed_equilibrium(optimized_a)

    uniform_compliance = float(f @ uniform_u)
    optimized_reference_compliance = float(f @ optimized_reference_u)
    return (
        optimized_a,
        optimized_reference_compliance,
        optimized_reference_u,
        uniform_a,
        uniform_compliance,
        uniform_u,
    )


@app.cell(hide_code=True)
def _(
    F,
    L,
    V_max,
    a_max,
    a_min,
    diagnostics,
    epsilon_target,
    f,
    mo,
    np,
    optimized_a,
    optimized_reference_compliance,
    optimized_reference_u,
    result,
    u,
    uniform_compliance,
):
    optimized_u = np.asarray(u.value, dtype=float)
    optimized_compliance = float(f @ optimized_u)
    optimized_volume = float(L @ optimized_a)
    improvement_percent = 100.0 * (uniform_compliance - optimized_compliance) / uniform_compliance

    assert result.succeeded, result.message
    assert np.isfinite(result.objective)
    assert result.final_epsilon <= epsilon_target
    assert result.residuals.max_violation <= 2e-5
    assert result.complementarity <= 2e-7
    assert abs(diagnostics.source_gap) <= 2e-7
    assert np.max(np.abs(optimized_u[F])) <= 1e-7
    assert np.min(optimized_a) >= a_min - 1e-7
    assert np.max(optimized_a) <= a_max + 1e-7
    assert optimized_volume <= V_max + 1e-7
    assert np.allclose(optimized_u, optimized_reference_u, atol=5e-4)
    assert abs(optimized_compliance - optimized_reference_compliance) <= 1e-3
    assert optimized_compliance < 0.85 * uniform_compliance

    mo.md(rf"""
    ## Result

    | quantity | value |
    | --- | ---: |
    | status | `{result.status}` |
    | uniform-design compliance | {uniform_compliance:.6f} |
    | optimized compliance | {optimized_compliance:.6f} |
    | compliance improvement | {improvement_percent:.2f}% |
    | material volume | {optimized_volume:.6f} / {V_max:.6f} |
    | optimized areas | {np.array2string(optimized_a, precision=3)} |
    | upper objective | {float(result.objective):.6f} |
    | final epsilon | {result.final_epsilon:.3e} |
    | maximum lifted residual | {result.residuals.max_violation:.3e} |
    | complementarity | {result.complementarity:.3e} |
    | signed source gap | {diagnostics.source_gap:.3e} |

    The optimized design reduces compliance by {improvement_percent:.1f}%
    without adding material. It thickens members along the dominant load path
    and leaves only the minimum area in members that contribute less stiffness
    for this loading direction. The independent equilibrium solve reproduces
    both the displacement and compliance returned by BLVPY.
    """)
    return (optimized_u,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load path and deformation

    The panels compare equal-volume designs. Line width represents member
    area, and color represents axial stress: blue members are in tension and
    red members are in compression. The light dashed lines show the undeformed
    truss, while the colored lines exaggerate displacement to reveal its mode.
    Loads and fixed supports are shown explicitly.
    """)
    return


@app.cell
def _(B, E, L, n_nodes, np, optimized_u, uniform_u):
    uniform_stress = E * (B @ uniform_u) / L
    optimized_stress = E * (B @ optimized_u) / L
    maximum_displacement = max(
        np.max(np.linalg.norm(uniform_u.reshape(n_nodes, 2), axis=1)),
        np.max(np.linalg.norm(optimized_u.reshape(n_nodes, 2), axis=1)),
    )
    deformation_scale = 0.32 / maximum_displacement
    stress_limit = max(np.max(np.abs(uniform_stress)), np.max(np.abs(optimized_stress)))
    return deformation_scale, optimized_stress, stress_limit, uniform_stress


@app.cell(hide_code=True)
def _(
    Path,
    deformation_scale,
    f,
    fixed_nodes,
    members,
    nodes,
    np,
    optimized_a,
    optimized_stress,
    optimized_u,
    plt,
    stress_limit,
    uniform_a,
    uniform_stress,
    uniform_u,
):
    truss_figure, truss_axes = plt.subplots(1, 2, figsize=(8.5, 3.4), layout="constrained")
    stress_norm = plt.Normalize(vmin=-stress_limit, vmax=stress_limit)
    stress_cmap = plt.get_cmap("coolwarm_r")
    maximum_area = max(np.max(uniform_a), np.max(optimized_a))

    def draw_truss(axis, a_value, u_value, member_stress, title):
        deformed_nodes = nodes + deformation_scale * u_value.reshape(nodes.shape)
        for start_node, end_node in members:
            axis.plot(
                nodes[[start_node, end_node], 0],
                nodes[[start_node, end_node], 1],
                color="0.78",
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )
        for member_index, (start_node, end_node) in enumerate(members):
            axis.plot(
                deformed_nodes[[start_node, end_node], 0],
                deformed_nodes[[start_node, end_node], 1],
                color=stress_cmap(stress_norm(member_stress[member_index])),
                linewidth=0.7 + 5.0 * a_value[member_index] / maximum_area,
                solid_capstyle="round",
                zorder=2,
            )

        axis.scatter(
            deformed_nodes[:, 0],
            deformed_nodes[:, 1],
            s=15,
            color="black",
            zorder=3,
        )
        axis.scatter(
            nodes[fixed_nodes, 0] - 0.07,
            nodes[fixed_nodes, 1],
            marker=">",
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )
        for node_index in (4, 5):
            load_y = f[2 * node_index + 1]
            loaded_node = deformed_nodes[node_index]
            arrow_length = 0.18 + 0.16 * abs(load_y)
            axis.annotate(
                "",
                xy=loaded_node,
                xytext=(loaded_node[0], loaded_node[1] + arrow_length),
                arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.1},
                zorder=5,
            )

        axis.set(
            xlim=(-0.2, 2.25),
            ylim=(-0.48, 1.28),
            aspect="equal",
        )
        axis.set_title(title, fontsize=15)
        axis.axis("off")

    draw_truss(
        truss_axes[0],
        uniform_a,
        uniform_u,
        uniform_stress,
        "Uniform sizing",
    )
    draw_truss(
        truss_axes[1],
        optimized_a,
        optimized_u,
        optimized_stress,
        "Optimized sizing",
    )

    stress_map = plt.cm.ScalarMappable(norm=stress_norm, cmap=stress_cmap)
    stress_colorbar = truss_figure.colorbar(
        stress_map,
        ax=truss_axes,
        orientation="horizontal",
        fraction=0.07,
        pad=0.02,
        shrink=0.65,
    )
    stress_colorbar.set_label("Axial stress (tension $>0$)", fontsize=13)
    stress_colorbar.ax.tick_params(labelsize=13)

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "planar_truss_sizing.pdf"
    truss_figure.savefig(figure_path, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
