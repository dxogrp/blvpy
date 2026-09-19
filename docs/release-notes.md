# Release notes

## 0.4

- Added {meth}`blvpy.BilevelProblem.polish`, which re-solves the lower problem
  at a result's fixed upper point and returns an immutable
  {class}`blvpy.PolishResult` with complete variable snapshots, feasibility,
  the polished upper objective, and a relative upper-objective improvement
  ratio. Polishing restores all model state and prints a concise summary by
  default.
- Added examples for the new {doc}`polishing` feature.
- Reorganized the examples into a published gallery and a repository-only
  advanced collection with shared notebook assets.

## 0.3

- Added `cp.Maximize` support at both bilevel levels. Lower maximization
  objectives are normalized to an equivalent minimization before conic
  canonicalization, while `LowerProblem.objective` and solve-time upper
  objective values remain in their original modeled sense.
- Made complete and partial best-of run selection choose the lowest upper
  objective for minimization and the highest for maximization. Defined
  `GapDiagnostics.source_gap` as the returned lower objective minus the
  reference optimum for minimization, and the reference optimum minus the
  returned lower objective for maximization.
- Add examples.

## 0.2

- Renamed the structural compatibility predicate to
  `BilevelProblem.is_dblp()` and standardized DBLP terminology throughout the
  public API and documentation.
- Added the Stackelberg port-security application notebook and a shared
  publication-oriented plotting style for the example gallery.

## 0.1

First public release of BLVPY.
