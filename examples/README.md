# Examples

BLVPY's examples are executable [Marimo](https://marimo.io/) notebooks organized into three collections:

- `gallery/` contains the focused examples executed and published with the documentation.
- `advanced/` contains repository-only workflows that demonstrate advanced usage or are too expensive for documentation builds. These notebooks are checked statically but are not executed by documentation or release workflows.
- `_shared/` contains assets shared by notebooks and is not itself a notebook collection.

Install the example dependencies and open the complete workspace with:

```shell
make sync-examples
make marimo
```

Before contributing or modifying a notebook, run the recursive strict check across both notebook collections:

```shell
make check-examples
```

Gallery notebooks should be deterministic, reasonably quick to execute, and suitable for a standalone published snapshot. Put specialized or computationally expensive workflows in `advanced/` instead.

## Advanced workflows

- [`polishing_workflow.py`](advanced/polishing_workflow.py) performs a deliberately
  coarse nonlinear solve, inspects a feasible polished lower response, and adopts
  that candidate explicitly.
