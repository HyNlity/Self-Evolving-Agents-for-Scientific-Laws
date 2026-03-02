# Evo Protocol: Complete Rules

## Philosophy

Scientific discovery is iterative. Each experiment is an investment that should yield knowledge, whether it succeeds or fails. The Evo Protocol ensures that every iteration builds on previous ones and that the search space is explored efficiently.

## The Core Loop

```
1. READ plan.md (understand current state)
2. HYPOTHESIZE (form testable prediction)
3. EXPERIMENT (design + execute)
4. RECORD (update plan.md with results)
5. REFLECT (was the hypothesis confirmed? what did we learn?)
6. ITERATE (back to step 1 with new knowledge)
```

## Attention Management

### Pre-Experiment Checklist
Before running any PySR call or analysis script:
- [ ] Read `plan.md` Confirmed Knowledge section
- [ ] Read `plan.md` Failed Approaches table
- [ ] Check: is your planned experiment different from all failed ones?
- [ ] Check: does your strategy build on confirmed knowledge?

### Post-Experiment Update
After receiving results:
- [ ] Update `Confirmed Knowledge` if the result is informative
- [ ] Add to `Failed Approaches` if the strategy didn't improve MSE
- [ ] Update `Current Hypotheses` based on new evidence
- [ ] Add next steps to `Strategy Queue`

## Strategy Mutation Rules

### Minimum Viable Difference
Each new experiment must change at least ONE of:
- **Variables**: different feature subset
- **Operators**: different unary/binary operators
- **Structure**: different expression template or maxsize
- **Parameters**: different parsimony, populations, niterations
- **Data**: derived features, filtered subset, transformed target

### Escalation Ladder
When incremental changes stop improving:
1. First: try different variable combinations
2. Then: try different operator sets
3. Then: try feature engineering (derived variables)
4. Then: try radically different approach (e.g., ODE discovery instead of static regression)
5. Finally: increase computational budget (more iterations, larger maxsize)

## Recording Standards

### Failed Approaches Table
Each entry must include:
- **Round**: when it was tried
- **Strategy**: brief description of what was attempted
- **Variables**: which variables were used
- **Template/Params**: expression template or key PySR parameters
- **MSE**: result metric
- **Why Failed**: brief explanation of why this didn't work

### Confirmed Knowledge
Update when:
- A variable is confirmed as relevant (appears in multiple good equations)
- A variable is confirmed as irrelevant (never improves results)
- A specific relationship is confirmed (e.g., "x1 and x3 interact multiplicatively")
- A specific operator is important (e.g., "sin() is needed for this data")

## Anti-Patterns to Avoid

1. **Strategy Loop**: repeating a failed approach with minor tweaks
2. **Kitchen Sink**: throwing all variables at PySR without hypotheses
3. **Complexity Creep**: always increasing maxsize without parsimony
4. **Tunnel Vision**: only trying one type of approach (e.g., only linear combinations)
5. **Amnesia**: not reading plan.md before designing experiments
