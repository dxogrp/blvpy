# BLVPY: Disciplined Bilevel Programming

BLVPY is a [CVXPY](https://www.cvxpy.org/) extension for modeling and locally
solving optimistic bilevel programs. The lower-level problem is written as a
disciplined parametrized program (DPP), with selected upper variables declared
as its parameters. BLVPY canonicalizes that family once and replaces
lower-level optimality with primal feasibility, dual feasibility, and a conic
duality-gap constraint.

For a canonical lower problem

$$
\mathop{\mathrm{minimize}}_u\;c(x)^T u+d(x)
\quad\text{subject to}\quad A(x)u+s=b(x),\;s\in K,
$$

the relaxed single-level problem enforces

$$
s^T\lambda\leq\epsilon
$$

together with primal and dual conic feasibility. A warm-started continuation
method then decreases $\epsilon$. When those feasibility conditions hold, the
gap bounds lower-level objective suboptimality by $\epsilon$.

> [!IMPORTANT]
> BLVPY uses a local nonlinear solver. A successful solve is not a certificate
> of global upper-level optimality. Inspect the reported feasibility and gap
> residuals; solver termination alone is not a bilevel certificate.

## Supported models

The initial release supports exact lower-problem canonicalizations containing
only:

- zero cones (equalities),
- nonnegative cones (linear inequalities), and
- second-order cones.

This includes LPs, losslessly conic-representable QPs, and SOCPs. Models that
canonicalize to positive-semidefinite, exponential, or power cones are rejected
before the nonlinear solve. Lower problems must be real, continuous,
minimization problems and DPP with respect to their linked parameters. BLVPY
implements optimistic semantics: if the lower problem has several minimizers,
the upper problem may select the one most favorable to it.

Ordinary CVXPY parameters may also occur as fixed data. Their values are
snapshotted when the lower problem is first canonicalized; construct a new
`BilevelProblem` if those values need to change.

For the paper's pointwise graph contract, the initial exact-atom audit accepts
affine expressions plus absolute value, Huber loss, scalar/vector maxima and
minima, 1/2/infinity norms, quadratic forms and squares,
quadratic-over-linear, and sum-largest graphs. A model may canonicalize to an
SOCP yet still be rejected when its source atom has not been audited; this
conservative policy avoids treating cone type alone as a losslessness proof.

## Installation

BLVPY requires Python 3.12 or newer, CVXPY 1.9 or newer, and IPOPT. Install
the native IPOPT library for your operating system first, then install BLVPY:

```shell
pip install blvpy
```

The required `cyipopt` Python binding is installed with BLVPY. It needs the
native IPOPT library to be present while BLVPY is installed and used.

IPOPT is BLVPY's mandatory, default, and natively tested nonlinear backend.
CVXPY 1.9 also exposes DNLP interfaces for KNITRO, UNO, and COPT. Those
alternatives may be selected when their Python packages, native libraries, and
any required licenses are installed independently; BLVPY does not install or
natively test them.

Canonicalization uses a fixed Clarabel-compatible reduction. Clarabel, which is
included with CVXPY, is also the default conic backend for upper projection and
fixed-upper lower initialization; `solve(conic_solver=...)` may select another
compatible backend for those numerical solves.

## Basic usage

The same CVXPY variable appears directly in both levels. Declare which upper
variables parameterize the lower problem with `LowerProblem`:

```python
import cvxpy as cp

from blvpy import BilevelProblem, LowerProblem

x = cp.Variable(name="x")
y = cp.Variable(name="y")

lower = LowerProblem(
    cp.Minimize(cp.square(y - x)),
    parameters=[x],
)
bilevel = BilevelProblem(
    outer_objective=cp.Minimize(cp.square(x - 1.0) + cp.square(y + 1.0)),
    lower_problem=lower,
)

assert bilevel.is_dbp()
result = bilevel.solve(
    epsilon_initial=1e-1,
    epsilon_target=1e-6,
)

print(result.status)
print(x.value, y.value)
print(result.residuals)
```

Every variable listed in `LowerProblem(parameters=...)` is an upper-level
variable and is held fixed while the lower problem is solved. Native CVXPY
`bounds=` are optional mathematical constraints and initialization hints;
BLVPY never requires bounds. BLVPY uses one deterministic upper point by
default: it preserves an existing `.value`, uses the midpoint of finite bounds,
uses `lower + 1` or `upper - 1` for a one-sided bound, and otherwise uses zero.
It then attempts to project that point onto DCP upper constraints.

Pass `best_of=N` to request randomized local search. BLVPY requests exactly
`N` sampled initialization attempts without discarding duplicates. Each viable
initialization runs a complete, independent epsilon continuation, and BLVPY
returns the acceptable target-epsilon result with the lowest upper objective.
Even `best_of=1` requests one random initialization; omitting `best_of` selects
the deterministic policy.

Set CVXPY's sampling-only `sample_bounds` for an upper variable that should be
randomized without constraining the mathematical problem:

```python
x.sample_bounds = (-2.0, 2.0)
result = bilevel.solve(best_of=5, seed=7)
```

For an explicit `best_of`, `sample_bounds` takes precedence over `.value`.
Otherwise an existing `.value` is reused in every run, or finite native
`bounds=` supply the sampling interval. BLVPY raises a named initialization
error when none of these is available. Sampling uses a solve-local random
generator, so a fixed `seed` reproduces the runs without changing NumPy's
global random state. Runtime scales approximately linearly with `best_of`
because every viable run performs its own full continuation.

`validate()` raises a detailed modeling error, while `is_dbp()` provides the
corresponding boolean check. `canonicalize()` exposes immutable cone-program
metadata for diagnostics. `solve()` returns the final source and canonical
variables, solver and continuation histories, complementarity, and separate
upper, recovery, primal, dual, cone, and gap residuals.

`result.epsilon_history` contains only accepted, strictly decreasing
tolerances for the selected run. `result.attempted_epsilon_history` also
includes its failed solves and retry points. `result.runs` preserves every
complete or failed run, while `result.selected_run` identifies the returned
one. Numerical NLP output remains local and residual-based; it is not a
rigorous finite-precision certificate.

Each `RunRecord.index` is zero-based, and `initial_values` snapshots the actual
post-projection upper point. `all_objectives` reports terminal objectives in run
order. If no run reaches the target, BLVPY returns `continuation_failed` at the
partial run that attained the smallest epsilon, with objective and run index as
tie-breakers.

## Gap diagnostics

Use the returned solution to request a complete primal-dual gap diagnosis:

```python
diagnostics = bilevel.gap_diagnostics(
    result,
    solver=cp.SCS,
    solver_options={"eps": 1e-8},
    solver_verbose=False,
)

print(diagnostics.source_gap)
print(diagnostics.identity_error)
```

This call performs one additional conic solve of the lower problem at the
returned upper point. Clarabel is the quiet default; `solver` may name another
CVXPY conic backend compatible with the lower problem. `solver_options` are
copied and passed through unchanged, while `solver_verbose` independently
controls its backend output.

`source_gap` is the signed difference between the returned lower objective and
that reference optimum, in the original lower objective's units. Small
negative values can occur within solver tolerance. The remaining fields
decompose the canonical primal-dual gap and check its inexact identity.
Pass an explicit solver name when overriding Clarabel; `solver=None` is not
supported.

Diagnosis is opt-in and does not change the result or model state. It provides
a numerical consistency check, not a certificate of global bilevel
optimality.

## Progress and solver output

`solve(solver=...)` selects the nonlinear backend used for feasibility
restoration and every continuation step. IPOPT is the default and the only
backend exercised by BLVPY's native integration suite. With independently
installed solver software, CVXPY 1.9 also accepts `cp.KNITRO`, `cp.UNO`, and
`cp.COPT` through its DNLP path:

```python
result = bilevel.solve(
    solver=cp.UNO,
    solver_options={"preset": "filtersqp"},
)
```

Solver-specific modes remain options: for example, UNO provides `filtersqp`
and `ipopt` presets, while KNITRO exposes algorithm choices through
`solver_options`. BLVPY copies and passes these options to CVXPY without
normalizing them. Users are responsible for installing each alternative's
Python package and native runtime, and for obtaining any required license.
Results from every backend retain the same local, non-certifying semantics.
BLVPY does not auto-select a nonlinear backend: pass an explicit solver name,
and do not use `solver=None`.

CVXPY also accepts the string aliases `"knitro_ipm"`, `"knitro_sqp"`,
`"knitro_alm"`, `"uno_ipm"`, and `"uno_sqp"`. These aliases select presets
for their base solver; they are not separate BLVPY backends.

BLVPY separates its concise progress transcript from CVXPY and native solver
output. Progress is enabled by default (`verbose=True`), while raw backend
output remains disabled by default (`solver_verbose=False`). The two controls
on `solve()` are independent:

| `verbose` | `solver_verbose` | Output |
|---|---|---|
| `False` | `False` | Quiet, with backend output suppressed on a best-effort basis |
| `True` | `False` | BLVPY progress only (default) |
| `False` | `True` | CVXPY and native solver output only |
| `True` | `True` | BLVPY progress and backend output |

For a readable account of a solve without the repeated backend transcripts,
use:

```python
result = bilevel.solve(
    verbose=True,
    solver_verbose=False,
)
```

A typical abbreviated continuation excerpt is:

```text
(BLVPY) Run 1/3 | begin | mode=random
(BLVPY) Run 1/3, attempt 1 [initial]: accepted | eps=1.000e-01
(BLVPY)   status=optimal
(BLVPY)   objective=5.003e-01 | feasibility=2.100e-08 | gap=0.000e+00
(BLVPY) Run 1/3: succeeded | status=optimal
(BLVPY) Selected run 1/3 | objective=5.000e-01
(BLVPY) Status: status=optimal | objective=5.000e-01
(BLVPY)   final_epsilon=1.000e-06
```

The BLVPY transcript is written to standard error and groups information into
`Problem`, `Initialization`, `Continuation`, and `Summary` sections. It reports
model dimensions and cone layout, search mode and run count, each run's
scheduled or retry epsilon, solver status and objective, available residuals,
solver time and iteration counts, terminal run outcomes, and the selected run.
It does not print sampled variable values or complete solver-option
dictionaries. The returned result and its run and iteration records remain the
machine-readable source of truth. Run and continuation records use a short
outcome row followed by indented diagnostic rows, and every BLVPY-owned line is
limited to 79 columns.

With IPOPT, `solver_verbose=False` adds quiet defaults (`print_level=0` and
`sb="yes"`) only when those keys are absent from `solver_options`. These
IPOPT-specific defaults are not injected into alternative backends. Explicit
solver options always take precedence, so a user-supplied `print_level` or `sb`
may intentionally produce output even when `solver_verbose` is false. Silence
is best-effort because some native-library messages are emitted below CVXPY's
logging controls.

## Examples

The [`examples`](examples) directory contains:

- [`analytic_quadratic.py`](examples/analytic_quadratic.py), whose exact
  bilevel solution is known;
- [`optimistic_lp.py`](examples/optimistic_lp.py), illustrating selection from
  a set of lower-level minimizers;
- [`parameter_dependent_socp.py`](examples/parameter_dependent_socp.py), with
  an upper-dependent constraint matrix and a second-order cone;
- [`best_of_restoration.py`](examples/best_of_restoration.py), showing
  sampling-only ranges, full best-of runs, and feasibility restoration; and
- [`continuation_reproducibility.py`](examples/continuation_reproducibility.py),
  exposing per-run tolerances, retries, and seeded reproducibility.

Run an example with:

```shell
uv run python examples/analytic_quadratic.py
```

## Development

BLVPY is currently a prototype. Until its API is declared stable, an API change
replaces the earlier form immediately unless the change request explicitly
requires otherwise.

BLVPY follows the `uv` and Hatchling workflow used by the other `dxogrp`
CVXPY extensions:

```shell
make sync       # create the environment and install development dependencies
make test       # run the full test suite
make lint       # run Ruff without modifying files
make build      # build the source and wheel distributions
```

CI runs the full test suite, lint checks, and package builds on Linux, macOS,
and Windows for Python 3.12 through 3.14. Linux and macOS use `uv` with their
native IPOPT installations. Windows uses one Miniforge environment for the
conda-forge IPOPT, `cyipopt`, and OpenBLAS packages; `uv` installs BLVPY and its
locked Python development dependencies directly into that environment and
builds the package artifacts.

## License

BLVPY is licensed under the [Apache License 2.0](LICENSE).
