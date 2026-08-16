# Solving

## Epsilon continuation

The canonicalized lower problem is a convex conic program in the form

$$
\begin{array}{ll}
\text{minimize} & c(x)^T u + d(x)\\
\text{subject to} & A(x)u + s = b(x)\\
& s\in \mathcal{K},
\end{array}
$$

where $u, s$ are the lower primal and slack variables, $c(x), d(x), A(x), b(x)$ are upper-dependent canonical data, and $\mathcal{K}$ is a Cartesian product of cones.
BLVPY solves the bilevel problem by continuation on a small relaxation of the lower KKT complementarity condition.

BLVPY introduces a dual vector $\lambda\in \mathcal{K}^*$ and imposes

$$
A(x)^T\lambda+c(x)=0,
\qquad s^T\lambda\leq\epsilon.
$$

The default solve begins at $\epsilon=10^{-1}$ and contracts it by $0.1$ until $10^{-6}$.
Each accepted point warm-starts the next DNLP solve.
If a scheduled step fails, BLVPY can retry from an intermediate epsilon, subject to `max_retries`.

An example of a solve call is

```python
result = problem.solve(
    epsilon_initial=1e-1,
    epsilon_target=1e-7,
    contraction=0.1,
    feasibility_tolerance=1e-7,
    max_retries=8,
)
```

## Deterministic initialization and restoration

The ordinary `solve()` path uses one deterministic upper point.
If an upper variable already has a `.value`, BLVPY preserves it. Otherwise,
for each scalar component $x_i$ with native lower and upper bounds
$l_i$ and $u_i$, it constructs

$$
\widetilde{x}_i=
\begin{cases}
(l_i+u_i)/2,
  & l_i,u_i\text{ are finite},\\
l_i+1,
  & l_i\text{ is finite and }u_i=+\infty,\\
u_i-1,
  & l_i=-\infty\text{ and }u_i\text{ is finite},\\
0,
  & l_i=-\infty\text{ and }u_i=+\infty.
\end{cases}
$$

BLVPY first projects $\widetilde{x}$ through the variable's CVXPY attributes,
such as nonnegativity or symmetry, to obtain $\widehat{x}$. When the upper and
generated linked-variable domain constraints define a DCP set $\mathcal{U}$,
BLVPY then attempts the least-distance projection

$$
x^{(0)}\in\mathop{\mathrm{argmin}}_{x\in\mathcal{U}}
  \sum_v\left\|x_v-\widehat{x}_v\right\|_F^2,
$$

where the sum is over the upper variables. This projection is best effort: if
it cannot be compiled or solved, BLVPY retains the attribute-projected point.

At $x^{(0)}$, the selected conic backend solves the fixed-upper canonical
lower problem

$$
\begin{array}{ll}
\text{minimize} & c(x^{(0)})^T u \\
\text{subject to} & A(x^{(0)})u+s=b(x^{(0)})\\
  & s\in\mathcal{K}
\end{array}
$$

with variables $u, s$.
The conic solution supplies the initial canonical primal $u^{(0)}$, slack
$s^{(0)}$, and equality dual $\lambda^{(0)}$; BLVPY recovers the corresponding
source lower variables from its affine recovery map.

If every automatic path fails, BLVPY raises an `InitializationError` naming variables for which an explicit `.value` may help.

When `restoration=True` (the default) and this initial point fails BLVPY's
independent residual check, BLVPY introduces a nonnegative restoration radius
$\rho$ and schematically solves

$$
\begin{array}{ll}
\mathop{\mathrm{minimize}} & \rho\\
\mathop{\mathrm{subject\ to}}
  & \text{each lifted constraint is relaxed by }\rho,\\
  & s^T\lambda\leq\epsilon+\rho,\\
  & \rho\geq0.
\end{array}
$$

Restoration seeks a compatible feasible starting point; it does not optimize
the upper objective. BLVPY proceeds only if the recomputed feasibility and
relaxed-gap residuals are within `feasibility_tolerance`.

(best-of-search)=
## Best-of search for local solutions

With `best_of=None`, BLVPY follows one deterministic path.
Explicit `best_of=N`, including `N=1`, generates exactly $N$ upper initializations and runs a complete, independent continuation for every viable one.
Eligible components are sampled; components controlled by an existing `.value` remain fixed unless `sample_bounds` overrides that value.
Only runs that reach the target epsilon with acceptable residuals compete, and BLVPY selects the lowest final upper objective, breaking ties by run index.

```python
x.sample_bounds = (-2.0, 2.0)  # sampling metadata (not a constraint)
result = problem.solve(best_of=5, seed=42)

print(result.selected_run_index)
print(result.all_objectives)
for run in result.runs:
    print(run.index, run.initial_values[x], run.epsilon_history)
```

Sampling precedence is:

1. finite `sample_bounds`, which override `.value` for that variable;
2. an existing `.value`, reused in every run;
3. finite two-sided native CVXPY bounds; otherwise
4. {class}`blvpy.InitializationError`, naming the variables that need sampling information.

`sample_bounds` must be a finite `(lower, upper)` pair broadcastable to the variable shape.

## Solvers and options

```python
result = problem.solve(
    solver=cp.IPOPT,
    conic_solver=cp.CLARABEL,
    solver_options={"max_iter": 500},
    conic_solver_options={"max_iter": 200},
)
```

BLVPY copies the option mappings and forwards them to the corresponding CVXPY solve.
It uses the selected DNLP backend consistently for restoration and continuation.
It uses the selected conic backend for initialization and upper projection.
Availability is checked when CVXPY actually invokes the backend.

`verbose=True` (the default) prints concise BLVPY progress to standard error.
`solver_verbose=False` (the default) suppresses CVXPY and native solver output on a best-effort basis.
The flags are independent.
For quiet IPOPT calls, BLVPY supplies `print_level=0` and `sb="yes"` only when the user did not provide those options.
