"""Optimistic selection from a lower LP with many optimal solutions.

Requires BLVPy's native IPOPT dependency.
"""

import cvxpy as cp

from blvpy import BilevelProblem


def main() -> None:
    x = cp.Variable(name="x", bounds=[0.0, 1.0])
    y = cp.Variable(name="y")
    x_lower = cp.Parameter(name="x_lower")

    # Every y in [x, 1] is lower-optimal. Optimistic semantics lets the upper
    # problem choose y = 1, while its x^2 term selects x = 0.
    lower = cp.Problem(cp.Minimize(0.0), [y >= x_lower, y <= 1.0])
    problem = BilevelProblem(
        outer_objective=cp.Minimize(cp.square(x) + cp.square(y - 1.0)),
        lower_problem=lower,
        parameter_map={x_lower: x},
    )

    result = problem.solve(starts=5, seed=0)
    print(f"status: {result.status}")
    print(f"x: {x.value:.6f} (expected 0)")
    print(f"y: {y.value:.6f} (expected 1 under optimistic semantics)")
    print(f"complementarity: {result.complementarity:.3e}")


if __name__ == "__main__":
    main()
