# Retail Sales Patterns and Prediction — CRISP-DM Summary

**Project:** Retail Sales Patterns and Prediction using FreshRetailNet-50K
**Course:** MIS480-50 Predictive Analytics · Dr. Aindrila Chakraborty · Fall 2026
**Team:** Paul Ries, Feifei Li, Sem Masgina, Derrick Nyagesuka

This one-page summary maps the group's data-preparation, exploration, and modeling work
to the six phases of **CRISP-DM** (Cross-Industry Standard Process for Data Mining).
Work lives in five notebooks: `01_inspect`, `02_subset`, `03_eda`,
`04_modeling_and_evaluation` (modeling + evaluation), `05_classification`.

---

## Phase 1 — Business Understanding
**Research question:** *Which promotional, calendar, weather, and stockout factors are
associated with observed retail sales, and can information available in advance improve
next-day sales predictions beyond using past sales alone?* Findings are framed as
**associations and predictive contributions**, not proof of causation.

## Phase 2 — Data Understanding  *(notebook 01)*
- **Dataset:** FreshRetailNet-50K (Dingdong-Inc, Hugging Face) — **4.5M training + 350K
  evaluation rows**. The "50K" is ~**50,000 store-product series** (4.5M / 90 days), each
  a product in a store over 90 daily records, with scaled sales, hourly sales & stock
  arrays, promotion/holiday flags, a discount ratio, stockout hours, and weather.
- **Levels of measurement (Week 6):** ids are **nominal** (codes, not numbers), `dt` and
  temperature are **interval**, sales/discount/counts/weather are **ratio**.
- **Quality:** the working subset has **0 missing, 0 duplicate records, every series
  complete at 90/90 days**.

## Phase 3 — Data Preparation  *(notebook 02)*
- **Select:** 2 cities × 5 stores × 10 products × 90 days = **100 complete series /
  9,000 rows** (2024-03-28 → 2024-06-25); full histories kept, not isolated days. This is
  a documented **convenience sample**, not a random draw.
- **Clean:** verified counts; zero-sales and stockout rows preserved as real signal.
- **Construct:** engineered `availability = (16 − stockout_hours)/16` and
  `est_demand = sale_amount / availability` — a transparent, assumption-based proxy for
  the demand censored during stockouts.
- **Missing-data note (Week 6):** no values are missing, but stockout censoring is an
  **MNAR** pattern (unrecorded demand depends on its own size), which is *non-ignorable*
  — hence `est_demand`, treated as a proxy while observed `sale_amount` stays primary.

## Phases 2–3 — Exploratory Data Analysis  *(notebook 03)*
- **Conditions:** sales are higher on promotions, discounts, and holidays (descriptive).
- **Stockout censoring:** mean *observed* sales fall as stockout hours rise
  (1.14 → 0.46), but **stockout-adjusted demand rises** (1.14 → 4.72) — the decline is
  mostly censoring; high-demand items are the ones that sell out.
- **Weather is weak:** correlations with sales are near zero (temperature strongest at
  +0.07).
- **Store/series variation:** store means span ~0.9–1.7; the 100 series range from ~0.3
  to ~7.1 average daily sales (right-skewed) — real, usable variation.
- **Feature screen (Weeks 7–8):** most relevant predictors are `discount` (r ≈ −0.21) and
  `holiday_flag` (r ≈ +0.17); `stock_hour6_22_cnt` flips from −0.08 (observed) to +0.26
  (estimated demand). All **VIF < 3** → no multicollinearity (only `availability` ≡
  `stock_hour6_22_cnt` are identical by construction).
- **Weekly seasonality (Weeks 3–4 decomposition):** weekend sales (Sat/Sun ≈ 1.6) run
  about 50% above weekdays (≈ 1.0–1.1); the 90-day trend is flat. This weekly cycle is
  what the `lag_7` feature captures. (90 days cannot show annual seasonality.)

## Phase 4 — Modeling  *(notebook 04 — Modeling & Evaluation)*
- **Task:** predict next-day `sale_amount` using only information known at forecast time
  — past-sales lags (`lag_1`, `lag_7`, 7-day rolling mean), planned promotion/holiday/
  discount, and **lagged** weather/stockout. Target-day sales, stockout, and weather are
  excluded (they would leak).
- **Design:** chronological train/validation/test split; preprocessing fit on training
  only; numeric target → **Regression Trees + Multiple Linear Regression** (Week 8), not
  classifiers.
- **Within-sample test results** (fixed-date split, test Jun 12–25; Random Forest tuned to
  depth 10 / leaf 20 / 300 trees):

  | Model | MAE | RMSE | WAPE |
  |---|---|---|---|
  | baseline (previous-week same-day) | 0.545 | 0.844 | 45.1% |
  | linear regression | 0.455 | 0.671 | 37.6% |
  | **random forest** | **0.438** | **0.663** | **36.3%** |

  Excluding the 107 long-stockout rows (≥12h out of stock), Random Forest WAPE improves to **32.7%**.
- **Feature importance (permutation):** recent 7-day sales dominate; holidays and discounts
  are the top factors beyond past sales; weather adds almost nothing.
- **Diagnostics (same notebook):** residual/assumption checks (heteroscedasticity corr ≈ 0.52),
  95% **prediction intervals** (ŷ ± 1.96σ, well-calibrated at ~95% test coverage), and a short
  forecasting view (recursive multi-step + Holt-Winters). A `log1p` transform was explored
  (Week 8) but did not change the ranking, so the raw-target model is kept.
- **Forecasting check:** a recursive 7-day forecast run across all 100 series beats a
  hold-last-value persistence baseline (**MAE 0.517 vs 0.589**); on the smooth 100-series
  aggregate, Holt-Winters and a seasonal-naive "repeat last week" are essentially tied
  (MAE ≈ 0.10), confirming a weekly but no annual cycle to exploit.

## Phase 5 — Evaluation  *(notebook 04)*
Scored once on the dataset's **official eval split** — the true next 7 days
(2024-06-26 → 2024-07-02) for the same 100 series, with lags built across the
history→future boundary (no leakage).

| Model | MAE | RMSE | WAPE |
|---|---|---|---|
| baseline (previous-week same-day) | 0.601 | 0.960 | 48.8% |
| linear regression | 0.491 | **0.749** | 39.9% |
| **random forest** | **0.475** | 0.788 | 38.6% |

**Verdict:** every model beats the naive baseline on genuinely unseen days, so advance
information gives a **modest but real** improvement over past sales alone — the answer to
the research question. On the eval week the **Random Forest has the best MAE (0.475)** and
**linear the best RMSE (0.749)** — effectively tied, so a simple, interpretable model is
defensible.

## Secondary analysis — Classification (Week 9)  *(notebook 05)*
To exercise the Week 9 classification toolkit, we reframed the task as a yes/no question
— **"will tomorrow be a high-sales day?"** (next-day `sale_amount` above the training
median) — using the same no-leakage features and the same fixed-date split as the
modeling notebook (test = Jun 12–25). Logistic regression and a Random Forest
classifier were compared against a majority-class baseline and scored with the confusion
matrix, precision/recall, and ROC-AUC.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| majority baseline | 0.456 | — | — | — |
| logistic regression | 0.775 | 0.811 | 0.765 | 0.865 |
| random forest | 0.779 | 0.817 | 0.766 | 0.865 |

Both classifiers clearly beat the baseline (ROC-AUC ≈ 0.87). This is a **secondary
demonstration**: binarizing the target discards information, so the Phase 4 regression
remains the main forecasting model.

## Phase 6 — Deployment & Recommendations
We do not ship a production system; for this project, deployment means findings-based
recommendations and an operating plan.

- **Recommendation.** Use the tuned Random Forest (or the simpler linear model) to forecast
  next-day demand per store-product for inventory and promotion planning — it beats the
  previous-week baseline by about 20% on both the test set and the official eval week.
- **How it would run.** Retrain on all history; score each morning using **only information
  known at forecast time** (past-sales lags + planned promotion/holiday/discount + yesterday's
  weather and stockout); report each forecast **with its 95% prediction interval** so planners
  see the uncertainty.
- **Monitoring & maintenance.** Track MAE / RMSE / WAPE against realized sales; alert if error
  drifts above the baseline; retrain on a schedule (e.g., monthly) or when drift is detected.
- **Operational caveats (from our findings).** Forecasts are of **observed** sales (censored by
  stockouts), so pair them with stockout flags; the **90-day** window has no annual or
  holiday-season coverage, so re-validate before peak periods.

---

## Limitations (proposal §4)
- We predict **observed** sales, not full unmet demand; `est_demand` is a rough proxy.
- **The 90-day window is the dataset's full training span, not a sampling choice** —
  FreshRetailNet-50K provides exactly 90 train + 7 eval days per series (the whole 50,000-series
  dataset is only 90 days), so weekly but not annual seasonality can be modeled.
- The **100-series convenience sample** and anonymized ids limit generalization to the
  full chain.
