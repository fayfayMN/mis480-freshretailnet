# MIS 480 Modeling Results – Sem Masgina

**Goal:** predict next-day sales for each of the 100 store-product series, using only information known the day before.
**Split (2024):** train Apr 4–May 28 (the first 7 days are used only as history), validate May 29–Jun 11, test Jun 12–25.

## Test results (lower is better)

| Model | MAE | RMSE | WAPE |
|---|---|---|---|
| Baseline (same day last week) | 0.545 | 0.844 | 45.1% |
| Linear regression | 0.455 | 0.671 | 37.6% |
| **Random Forest** | **0.438** | **0.663** | **36.3%** |

## Key findings
- **Random Forest is the best model.** Its MAE is about 20% lower than the baseline. Linear regression is about 17% lower.
- **Past sales do most of the work.** A Random Forest using only past sales and day of week scored almost the same (MAE 0.439 vs. 0.438).
- **Holidays and discounts are the most useful factors beyond past sales.** Previous-day weather and stockout hours added almost nothing.
- **Long stockouts:** 107 of the 1,400 test rows had 12+ stockout hours. Without them, Random Forest WAPE improves to 32.7%, because low sales on those days come from empty shelves, not low demand.

## Limits
Sales are scaled (relative, not revenue), the data covers only 90 days, and the results show associations, not proof that a factor causes sales to change.

## Files
- `MIS480_Modeling_Sem.ipynb`: the full notebook (code, results, charts)
- `fig_model_comparison.png`, `fig_feature_importance.png`, `fig_actual_vs_predicted.png`: charts for the poster
- `model_results.csv`: the results table
