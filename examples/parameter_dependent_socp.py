import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A parameter-dependent second-order cone problem
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import marimo as mo
    import numpy as np

    from blvpy import BilevelProblem, LowerProblem

    return BilevelProblem, LowerProblem, cp, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The bilevel model

    The upper problem chooses a scalar $x$ in the interval $[0.5,2]$. For
    this fixed value, the lower problem selects $y\in\mathbf{R}^2$ and an
    epigraph variable $t\in\mathbf{R}$ by solving

    \[
    \begin{array}{ll}
    \underset{y,t}{\operatorname{minimize}}
        & t \\
    \operatorname{subject\ to}
        & \lVert y\rVert_2 \leq t, \\
        & x y_1+y_2\geq 1.
    \end{array}
    \]

    The upper problem anticipates this response and solves

    \[
    \underset{0.5\leq x\leq2}{\operatorname{minimize}}
       \quad (x-1.25)^2+0.1t^\star(x).
    \]

    Thus, $x$ is the upper variable, while $y$ and $t$ are the lower
    variables. The first upper-objective term prefers $x=1.25$; the second
    penalizes the minimum norm attained by the lower problem.

    In `LowerProblem`, `parameters=[x]` means that $x$ is held fixed when
    checking and solving the lower problem. For every fixed $x$, this is
    a convex SOCP, even though $xy_1$ would be bilinear if $x$ and $y$ were
    optimized jointly in an ordinary CVXPY problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Geometric interpretation of the lower problem

    Define

    \[
      a(x)=\begin{bmatrix}x\\1\end{bmatrix}.
    \]

    The moving-halfspace constraint is $a(x)^Ty\geq1$. Since minimizing $t$
    subject to $\lVert y\rVert_2\leq t$ is equivalent to minimizing
    $\lVert y\rVert_2$, the lower problem finds the minimum-norm point in

    \[
      \{y \mid a(x)^Ty\geq1\}.
    \]

    The optimum lies on the boundary $a(x)^Ty=1$. Cauchy--Schwarz gives

    \[
      1=a(x)^Ty\leq\lVert a(x)\rVert_2\lVert y\rVert_2,
    \]

    so $\lVert y\rVert_2\geq1/\lVert a(x)\rVert_2$. Equality holds when
    $y$ is parallel to $a(x)$. Therefore,

    \[
      y^\star(x)
      =\frac{a(x)}{\lVert a(x)\rVert_2^2}
      =\frac{1}{x^2+1}\begin{bmatrix}x\\1\end{bmatrix},
      \qquad
      t^\star(x)=\frac{1}{\sqrt{x^2+1}}.
    \]

    The notebook later uses these formulas as an independent numerical check
    of BLVPY's returned lower-level solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The analytically reduced upper problem

    Substituting the exact lower response reduces the bilevel problem to
    the one-dimensional problem

    \[
      \underset{0.5\leq x\leq2}{\operatorname{minimize}}
      \quad
      \phi(x)=(x-1.25)^2+\frac{0.1}{\sqrt{x^2+1}}.
    \]

    At an interior solution, the first-order condition is

    \[
      \phi'(x)
      =2(x-1.25)-\frac{0.1x}{(x^2+1)^{3/2}}=0.
    \]

    Solving this scalar equation gives the reference values

    \[
    \begin{aligned}
      x^\star &\approx 1.265084,\\
      y^\star &\approx (0.486489,\ 0.384551),\\
      t^\star &\approx 0.620121,\\
      \phi(x^\star) &\approx 0.062240.
    \end{aligned}
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specify and inspect the model

    CVXPY's norm epigraph becomes a second-order cone block. BLVPY extracts
    affine maps for the complete canonical data, so those data can be
    evaluated at different upper points without re-canonicalizing the model.
    """)
    return


@app.cell
def _(BilevelProblem, LowerProblem, cp):
    x = cp.Variable(name="x")
    y = cp.Variable(2, name="y")
    t = cp.Variable(name="t")

    lower = LowerProblem(
        cp.Minimize(t),
        [cp.norm(y, 2) <= t, x * y[0] + y[1] >= 1.0],
        parameters=[x],
    )
    problem = BilevelProblem(
        cp.Minimize(cp.square(x - 1.25) + 0.1 * t),
        lower,
        outer_constraints=[x >= 0.5, x <= 2.0],
    )
    return problem, t, x, y


@app.cell
def _(np, problem, x):
    # Canonicalize once. The returned metadata stores affine maps for the
    # conic data, so changing x only reevaluates those maps; it does not run
    # CVXPY's canonicalization chain again.
    canonical = problem.canonicalize()

    # Evaluate the canonical constraint matrix at two upper-variable values.
    x.value = 0.75
    data_at_low_x = canonical.apply_numeric()
    x.value = 1.50
    data_at_high_x = canonical.apply_numeric()

    # Clear the inspection value so BLVPY applies its normal initialization
    # policy when the bilevel problem is solved below.
    x.value = None

    # These checks confirm both features illustrated by the example: the
    # lower canonicalization contains an SOC block, and A changes with x.
    canonical_matrix_change = np.linalg.norm(data_at_high_x.A.toarray() - data_at_low_x.A.toarray())
    assert canonical.cone_layout.second_order, "Expected the lower problem to contain an SOC block."
    assert canonical_matrix_change > 0.1, "Expected the canonical constraint matrix to depend on x."
    return (canonical_matrix_change,)


@app.cell
def _(np, problem, t, x, y):
    epsilon_target = 1e-6
    result = problem.solve(epsilon_target=epsilon_target, verbose=True)
    diagnostics = problem.gap_diagnostics(result)

    reference_x = 1.265084110259083
    reference_direction = np.array([reference_x, 1.0])
    reference_y = reference_direction / (reference_direction @ reference_direction)
    reference_t = 1.0 / np.linalg.norm(reference_direction)
    reference_objective = (reference_x - 1.25) ** 2 + 0.1 * reference_t

    assert result.succeeded, result.message
    np.testing.assert_allclose(
        [float(np.asarray(x.value)), *np.asarray(y.value), float(np.asarray(t.value))],
        [reference_x, *reference_y, reference_t],
        atol=3e-3,
        rtol=0.0,
    )
    assert abs(result.objective - reference_objective) <= 5e-3, "Upper objective does not match the reference value."
    return diagnostics, epsilon_target, result


@app.cell(hide_code=True)
def _(
    canonical_matrix_change,
    diagnostics,
    epsilon_target,
    mo,
    result,
    t,
    x,
    y,
):
    mo.md(rf"""
    ## Result and interpretation

    - Status: `{result.status}`
    - Target relaxation: $\epsilon_{{\mathrm{{target}}}}={epsilon_target:.1e}$
    - Final epsilon: ${result.final_epsilon:.1e}$
    - Upper variable: $x={float(x.value):.6f}$
    - Lower vector: $y=({y.value[0]:.6f}, {y.value[1]:.6f})$
    - Lower epigraph variable: $t={float(t.value):.6f}$
    - Upper objective: ${result.objective:.6f}$
    - Change in canonical $A$ between two inspected points:
      ${canonical_matrix_change:.3e}$
    - Maximum lifted violation: ${result.residuals.max_violation:.3e}$
    - Complementarity: ${result.complementarity:.3e}$
    - Independently evaluated lower source gap: ${diagnostics.source_gap:.3e}$

    The returned lower point agrees with the projection formula above.
    """)
    return


if __name__ == "__main__":
    app.run()
