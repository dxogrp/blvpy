"""An SOCP whose constraint matrix depends on the upper variable.

Requires BLVPY's native IPOPT dependency.
"""

import cvxpy as cp

from blvpy import BilevelProblem, LowerProblem


def main() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(2, name="y")
    t = cp.Variable(name="t")

    # The row [x, 1] in the lower linear inequality varies with the upper
    # variable. The norm epigraph contributes a second-order cone block.
    lower = LowerProblem(
        cp.Minimize(t),
        [cp.norm(y, 2) <= t, x * y[0] + y[1] >= 1.0],
        parameters=[x],
    )
    problem = BilevelProblem(
        outer_objective=cp.Minimize(cp.square(x - 1.25) + 0.1 * t),
        lower_problem=lower,
        outer_constraints=[x >= 0.5, x <= 2.0],
    )

    result = problem.solve()
    print(f"status: {result.status}")
    print(f"x: {x.value:.6f}")
    print(f"y: {y.value}")
    print(f"t: {t.value:.6f}")
    print(f"complementarity: {result.complementarity:.3e}")


if __name__ == "__main__":
    main()
