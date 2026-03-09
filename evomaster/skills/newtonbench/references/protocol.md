# NewtonBench Protocol Notes

This reference summarizes the minimum protocol alignment when running NewtonBench from Hamilton.

## Core Constraints

1. Up to 10 interaction rounds per task.
2. At most 20 parameter sets per experiment request.
3. Final answer must include a Python function:

```python
def discovered_law(...):
    ...
```

4. Evaluate with two metrics:
- Symbolic Accuracy (equivalence judge)
- RMSLE (data fidelity)

## Recommended Hamilton Mapping

1. Keep single-agent loop in Hamilton.
2. Use `newtonbench` skill scripts as environment bridge.
3. Keep benchmark logic outside core agent code when possible.
4. Keep final law inside `finish.message` in a `<final_law>...</final_law>` block for downstream parsing.

