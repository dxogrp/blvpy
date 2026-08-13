"""Optimistic selection from a lower LP with many optimal solutions.

Requires BLVPy's native IPOPT dependency.
"""

import cvxpy as cp

from blvpy import BilevelProblem, LowerProblem


def main() -> None:
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    # Every y in [x, 1] is lower-optimal. Optimistic semantics lets the upper
    # problem choose y = 1, while its x^2 term selects x = 0.
    lower = LowerProblem(cp.Minimize(0.0), [y >= x, y <= 1.0], parameters=[x])
    problem = BilevelProblem(
        outer_objective=cp.Minimize(cp.square(x) + cp.square(y - 1.0)),
        lower_problem=lower,
        outer_constraints=[x >= 0.0, x <= 1.0],
    )

    result = problem.solve()
    print(f"status: {result.status}")
    print(f"x: {x.value:.6f} (expected 0)")
    print(f"y: {y.value:.6f} (expected 1 under optimistic semantics)")
    print(f"complementarity: {result.complementarity:.3e}")


if __name__ == "__main__":
    main()
