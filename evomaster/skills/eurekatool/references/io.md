# IO and Imports

## Files (Hamilton workspace)

All paths below are relative to the **Hamilton workspace** (the current working directory of the agents):

- `data.csv` : the dataset
- `experiment.json` : PySR parameters + results per round
- `analysis.md` : Hamilton analysis history (natural language)
- `insight.md` : Eureka findings (natural language, minimal template)
- `history/roundN/` : per-round scripts and results

## Code reuse (no use_skill required)

The skill is a layered manual. For reuse, import the toolbox in Python code you run in the workspace:

```python
from skills.eurekatool import tool
```

If you run a script from `history/roundN/scripts/`, it still runs with cwd=workspace, so the import above works the same.

