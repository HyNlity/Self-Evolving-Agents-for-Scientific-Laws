# Convergence Guide

## When to Declare Satisfaction

A symbolic regression task can be considered converged when ANY of:

### 1. MSE Threshold Met
- The current best equation achieves MSE below a task-specified threshold
- If no threshold is specified, use domain-appropriate judgment

### 2. MSE Plateau
- MSE has not improved by more than 1% for the last 3 rounds
- AND at least 3 different strategies have been tried
- This suggests the search space has been adequately explored

### 3. Equation Stability
- The same (or equivalent) equation appears as best for 2+ consecutive rounds
- Different variable subsets and parameter settings converge to the same form
- This is strong evidence of a true underlying relationship

### 4. Physical Interpretability
- The best equation has a clear physical/scientific interpretation
- It uses a reasonable number of variables (parsimony)
- It generalizes to OOD data (if available)

## When NOT to Stop

Do not declare convergence if:
- Only 1-2 strategies have been tried
- The current best MSE is clearly improvable (large residual patterns visible)
- Feature engineering hasn't been attempted and residuals show systematic patterns
- The equation is overly complex (high complexity score) and simpler alternatives haven't been explored

## Convergence Signals from Data

### Residual Analysis
- **Random residuals** → good fit, consider stopping
- **Systematic patterns** → missing terms, keep iterating
- **Outlier-driven MSE** → might need robust fitting or data cleaning

### Complexity-MSE Tradeoff
- Plot (or inspect) the Pareto front of complexity vs. MSE
- Look for an "elbow" where more complexity gives diminishing returns
- The equation at the elbow is often the best answer

### OOD Performance
- If `data_ood.csv` is available, check OOD MSE
- Large gap between in-sample and OOD MSE → overfitting
- Similar in-sample and OOD MSE → good generalization, consider stopping

## Progress Indicators

Use these to assess whether iteration is productive:

| Indicator | Good Sign | Bad Sign |
|-----------|-----------|----------|
| MSE trend | Decreasing across rounds | Flat or oscillating |
| Variable set | Converging to a stable subset | Changing randomly |
| Equation form | Consistent structure across rounds | Completely different each time |
| Residual patterns | Becoming more random | Persistent systematic patterns |
| Strategy diversity | Many different approaches tried | Repeating same approach |
