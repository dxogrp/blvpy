# Troubleshooting

## Validation fails before solving

Call `problem.validate()` directly instead of relying only on
`problem.is_dblp()`. The former preserves the detailed exception. Common causes
are a non-DPP lower expression, an objective with curvature incompatible with
its sense (such as maximizing a convex expression), an unsupported cone, an
approximate or unaudited atom, an unset ordinary CVXPY parameter, or a lifted
expression that CVXPY does not recognize as DNLP.

{class}`blvpy.ValidationError` and its subclasses identify structural model
issues. {class}`blvpy.CanonicalizationError` indicates that CVXPY did not expose
the exact canonical form BLVPY expected.

## A lower model is rejected

BLVPY validates the source expression and its canonical cone program as two
separate layers. The exception identifies which layer failed.

### Source-expression checks

- {class}`blvpy.UnsupportedModelError` means a source atom is outside the
  audited set. {doc}`supported-atoms` lists every directly audited nonlinear
  atom.
- {class}`blvpy.ApproximateCanonicalizationError` means an atom has nonzero or
  nonfinite approximation error, or a constraint uses a quadrature
  approximation. Accepted rational representations must report finite
  `approx_error` equal to zero. `PowCone3DApprox` is rejected because it uses
  an SOC approximation; use the native `cp.PowCone3D` constraint when the
  model requires an exact three-dimensional power cone.

### Canonical-form checks

At the cone level, BLVPY supports zero, nonnegative, second-order,
exponential, and three-dimensional power-cone blocks. PSD and N-dimensional
power-cone blocks are unsupported.

- {class}`blvpy.UnsupportedConeError` reports an unsupported cone block. Exact
  `cp.geo_mean` and direct `cp.PowConeND` constraints, for example, produce
  N-dimensional power cones.
- {class}`blvpy.CanonicalizationError` means CVXPY selected a reduction chain
  that BLVPY has not audited.

## IPOPT cannot be loaded

{class}`blvpy.SolverUnavailableError` means CVXPY could not load the requested
backend. For IPOPT, verify both the native library and `cyipopt` in the same
Python environment that runs BLVPY. See {doc}`installation`; calling
`cp.installed_solvers()` is not BLVPY's availability test because native
loading can still fail at solve time.

## Automatic initialization fails

In deterministic mode, assign `.value` to every variable named by the
{class}`blvpy.InitializationError`, then solve again. For explicit best-of
searches, each named upper variable needs finite `sample_bounds`, an existing
`.value`, or finite two-sided native bounds. See {ref}`best-of-search` for the
precedence rules.

An initialized upper point can still lead to an infeasible or unbounded fixed
lower problem. Read exception notes and the BLVPY progress transcript for the
conic status or restoration reason.

## Continuation does not reach the target

A returned `continuation_failed` result contains the best partial run: smallest
attained epsilon, then best finite objective in the modeled sense (lowest for
`cp.Minimize`, highest for `cp.Maximize`), then lowest run index. A missing or
nonfinite objective ranks after a finite objective at the same epsilon.
Inspect `result.runs`, `attempted_epsilon_history`, and each iteration's
message and residuals. Possible responses include a looser target, gentler
contraction, more retries, better scaling, explicit initialization, or a
best-of search.

## Nonlinear cone residuals are unexpectedly large

Exponential- and 3D power-cone residual distances are numerical projection
estimates. Scale a cone triple to a moderate common magnitude and, when an
equivalent formulation permits it, avoid extreme ratios between its
components. BLVPY removes a shared power-of-two scale exactly. Estimates near
solver resolution are retried or replaced by a conservative upper bound
rather than reported as zero.

If BLVPY cannot obtain a validated projection, it reports the distance to the
cone's zero element as a conservative upper bound. This fail-closed result can
cause an otherwise acceptable iterate to fail its residual check. See
{ref}`nonlinear-cone-distance-estimates` for the numerical contract.

## Diagnostics fail

`gap_diagnostics()` requires complete source and canonical snapshots and a
successful or `continuation_failed` result. Its reference lower solve can fail
independently of the DNLP solve. Choose another compatible conic solver or pass
solver-specific options when appropriate.
