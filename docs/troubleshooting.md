# Troubleshooting

## Validation fails before solving

Call `problem.validate()` directly instead of relying only on
`problem.is_dbp()`. The former preserves the detailed exception. Common causes
are a non-DPP lower expression, a lower maximization problem, an unsupported
cone, an approximate or unaudited atom, an unset ordinary CVXPY parameter, or
a lifted expression that CVXPY does not recognize as DNLP.

{class}`blvpy.ValidationError` and its subclasses identify structural model
issues. {class}`blvpy.CanonicalizationError` indicates that CVXPY did not expose
the exact canonical form BLVPY expected.

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
attained epsilon, then lowest finite objective, then lowest run index. Inspect
`result.runs`, `attempted_epsilon_history`, and each iteration's message and
residuals. Possible responses include a looser target, gentler contraction,
more retries, better scaling, explicit initialization, or a best-of search.

## Diagnostics fail

`gap_diagnostics()` requires complete source and canonical snapshots and a
successful or `continuation_failed` result. Its reference lower solve can fail
independently of the DNLP solve. Choose another compatible conic solver or pass
solver-specific options when appropriate.
