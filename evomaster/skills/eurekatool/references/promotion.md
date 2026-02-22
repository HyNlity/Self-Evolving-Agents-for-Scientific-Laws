# Promotion (Make It Reusable, Keep It Small)

Promote analysis into tools only if it is likely to be reused in later rounds.

Minimal promotion checklist:
- Add/extend a function in `tool.py` (keep old signatures stable).
- Add 1 line to `usage.txt` describing the new function/script.
- If the new feature needs explanation, append a short section to the most relevant reference file.

If the analysis is one-off, keep it in `history/roundN/scripts/` instead.

