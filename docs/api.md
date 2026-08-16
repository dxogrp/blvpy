# API reference

The symbols below are the complete public namespace exported by `blvpy` {sub-ref}`release`.
Objects in the advanced section expose canonical numerical metadata for inspection; their detailed structure might change.

## Modeling and solving

```{eval-rst}
.. autoclass:: blvpy.LowerProblem
   :members: objective, constraints, parameters
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.BilevelProblem
   :members: upper_variables, source_variables, is_dbp, validate, canonicalize, solve, gap_diagnostics
   :member-order: bysource
```

## Results and diagnostics

```{eval-rst}
.. autoclass:: blvpy.BilevelResult
   :members: epsilon_history, attempted_epsilon_history, solver_statuses, residuals, complementarity, final_epsilon, succeeded, selected_run, all_objectives
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.RunRecord
   :members: epsilon_history, attempted_epsilon_history, solver_statuses, residuals, complementarity, final_epsilon, succeeded
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.IterationRecord
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.Residuals
   :members: max_feasibility, max_violation, is_feasible, as_dict
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.GapDiagnostics
   :members: normalized_gap, inexact_identity_rhs, identity_error
   :member-order: bysource
```

## Exceptions

```{eval-rst}
.. autoexception:: blvpy.BilevelError
```

```{eval-rst}
.. autoexception:: blvpy.ValidationError
```

```{eval-rst}
.. autoexception:: blvpy.ParameterMappingError
```

```{eval-rst}
.. autoexception:: blvpy.UnsupportedModelError
```

```{eval-rst}
.. autoexception:: blvpy.UnsupportedConeError
```

```{eval-rst}
.. autoexception:: blvpy.ApproximateCanonicalizationError
```

```{eval-rst}
.. autoexception:: blvpy.CanonicalizationError
```

```{eval-rst}
.. autoexception:: blvpy.InitializationError
```

```{eval-rst}
.. autoexception:: blvpy.SolverUnavailableError
```

```{eval-rst}
.. autoexception:: blvpy.SolveError
```

## Advanced canonical inspection

:::{note}
These classes are public so that you can inspect the canonical lower problem:
for example, its cone layout, parameter-dependent numerical data, and source-
variable recovery maps. Obtain them by calling
{meth}`blvpy.BilevelProblem.canonicalize` and use their documented inspection
methods.

They are not intended to be constructed or modified by users. In particular,
do not instantiate these classes directly, mutate arrays or CVXPY expressions
stored inside them, or rely on their exact field organization remaining
unchanged.
:::

```{eval-rst}
.. autoclass:: blvpy.CanonicalLowerProblem
   :members: source_variable_ids, recovery_map, parameter_ids, apply_numeric, build_data_expressions, recovery_expressions, recover_numeric
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.CanonicalData
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.CanonicalExpressions
   :members:
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.ParameterSpec
   :members: pack_numeric, pack_expression
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.RecoverySpec
   :members: expression, numeric
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.AffineRecoveryMap
   :members: expressions, numeric
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.ConeLayout
   :members: from_dims, nonneg, soc, size, zero_slice, nonnegative_slice, nonneg_slice, second_order_slices, soc_slices, blocks, primal_constraints, dual_constraints, primal_distance, dual_distance, complementarity
   :member-order: bysource
```

```{eval-rst}
.. autoclass:: blvpy.ConeBlock
   :members: size, slice
   :member-order: bysource
```
