# Release notes

## 0.3

- Added `cp.Maximize` support at both bilevel levels. Lower maximization
  objectives are normalized to an equivalent minimization before conic
  canonicalization, while `LowerProblem.objective` and solve-time upper
  objective values remain in their original modeled sense.
- Made complete and partial best-of run selection objective-sense aware, and
  defined `GapDiagnostics.source_gap` as sense-normalized lower
  suboptimality.
- Add examples.

## 0.2

- Renamed the structural compatibility predicate to
  `BilevelProblem.is_dblp()` and standardized DBLP terminology throughout the
  public API and documentation.
- Added the Stackelberg port-security application notebook and a shared
  publication-oriented plotting style for the example gallery.

## 0.1

First public release of BLVPY.
