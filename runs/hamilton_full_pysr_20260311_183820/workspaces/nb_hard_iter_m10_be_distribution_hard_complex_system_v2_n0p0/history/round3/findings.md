### Round 3 Findings
Candidate:
```
def discovered_law(omega, T):
    import math
    C = 1.5e-21
    val = min(max(C*omega/T, -60), 60)
    return 1/(math.exp(val)-1)
```
Evaluation RMSLE: 31.77
Exact Accuracy: 0.0
Symbolic Equivalent: False
<!-- APPEND_FINDINGS -->