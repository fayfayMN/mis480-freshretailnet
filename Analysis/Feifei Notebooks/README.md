# MIS 480-50 — Retail Sales Patterns and Prediction

**Predicting next-day retail sales with the FreshRetailNet-50K dataset.**
A group project for **MIS 480-50 Predictive Analytics** · Metropolitan State University ·
Dr. Aindrila Chakraborty · Fall 2026.

**Team:** Paul Ries · Feifei Li · Sem Masgina · Derrick Nyagesuka

---

## Overview

**Research question:** *Which promotional, calendar, weather, and stockout factors are
associated with observed retail sales, and can information available in advance improve
next-day sales predictions beyond using past sales alone?*

The project follows the **CRISP-DM** process end to end — data understanding, preparation,
exploratory analysis, modeling, and evaluation — on a 100-series working subset of
FreshRetailNet-50K. Findings are reported as associations and predictive contributions,
not causal claims.

**Headline result (official held-out week):** every model beats a previous-week baseline;
advance information gives a modest but real improvement, and a tuned Random Forest
generalizes best (canonical numbers from `06_modeling_final`).

| Model | MAE | RMSE |
|---|---|---|
| baseline (previous-week same-day) | 0.601 | 0.960 |
| linear regression | 0.491 | 0.749 |
| random forest (tuned) | **0.475** | **0.788** |

See [`CRISP-DM_Summary.md`](CRISP-DM_Summary.md) for the full one-page write-up.

---

## Dataset

**FreshRetailNet-50K** (Dingdong-Inc) — a fresh-grocery retail dataset of ~50,000
store–product series (4.5M training + 350K evaluation daily records) with scaled sales,
hourly sales/stock-status arrays, promotion/holiday flags, a discount ratio, stockout
hours, and weather.

- **Source:** https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

> ⚠️ **The full raw files are not stored in this repo.** `Data/FreshRetailNet-50k/train.parquet`
> is ~101 MB (above GitHub's 100 MB limit) and the data is licensed upstream, so it is
> git-ignored. To obtain it:
> ```python
> from datasets import load_dataset
> ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")   # train + eval splits
> ```
> The small **derived subset** used by the analysis (100 series × 90 days) *is* included
> under `Data Prep - Derrick/data/subset/`, so the EDA, modeling, and classification
> notebooks run without downloading the full dataset.

---

## Repository structure

```
MIS480-50 Group Project/
├── README.md
├── Data/                              # dataset landing (raw parquet git-ignored; see above)
│   ├── FreshRetailNet.html            # dataset card
│   └── Dataset Shortlist.docx         # dataset selection notes
├── Data Prep - Derrick/               # data preparation, ETL, and notebooks
│   ├── 01_inspect … 06_modeling_final (.ipynb + .html)   # 06 = canonical team modeling
│   ├── scripts/                       # 01_inspect · 02_subset · 03_eda (.py)
│   ├── data/subset/                   # derived 100-series working subset (CSV + parquet)
│   └── outputs/                       # data_dictionary, quality_report, charts (eda/, models/)
└── Analysis/Feifei Notebooks/         # analysis copies + the write-up
    ├── 01_inspect … 05_classification (.ipynb + .html)
    └── CRISP-DM_Summary.md / .docx
```

Notebooks (run in order): **01** inspect → **02** subset → **03** EDA → **04** modeling →
**05** classification → **06** modeling (final, consolidated — the canonical modeling result).

---

## How to run

1. **Environment** (Python 3; conda or venv):
   ```bash
   pip install pandas numpy scikit-learn matplotlib pyarrow datasets jupyter
   ```
2. **Get the data:** the working subset is already in `Data Prep - Derrick/data/subset/`.
   To rebuild it from scratch, download the dataset (see **Dataset** above) and run the
   `scripts/` in order.
3. **Run the notebooks** from the folder that contains the `data/` and `outputs/` paths
   (`Data Prep - Derrick/`), or open the pre-rendered `.html` exports for a quick read.

---

## Results at a glance

- **EDA:** sales rise with promotions/discounts/holidays; weather is weak; strong **weekly
  seasonality** (weekends ~50% above weekdays); stockout censoring confirmed.
- **Modeling (canonical: `06_modeling_final`):** next-day forecast with no-leakage features
  (past-sales lags + planned promo/holiday/discount + lagged weather/stockout); previous-week
  baseline vs. linear vs. a **tuned Random Forest** (depth 10, leaf 20); reported with
  MAE/RMSE/WAPE, a long-stockout sensitivity check, and permutation importance. (`04_modeling`
  holds extra diagnostics — a `log1p` transform and 95% prediction intervals.)
- **Evaluation:** validated on the dataset's **official eval split** (true next 7 days).
- **Classification (Week 12):** a "high- vs low-demand day" demo — logistic regression and a
  Random Forest classifier reach ~0.79 accuracy, ROC-AUC ~0.87.

Charts: `Data Prep - Derrick/outputs/eda/` and `.../outputs/models/`.

---

## Project sections & ownership

**Paul Ries** acted as **project manager** — breaking the work into the sections below and
assigning owners (per the team's planning email):

| Section | Owner | Status |
|---|---|---|
| Dataset Overview | Feifei | ✅ Complete |
| Data Preparation | Derrick | 🔄 Needs update |
| EDA Results | Derrick | 🔄 Needs update |
| Modeling | Sem | 🔄 Needs update |
| Evaluation Metrics | Sem | 🔄 Needs update |
| Presentation Slides | All | ⬜ Not started |
| Final Report | All | ⬜ Not started |

> Draft notebooks, charts, and the CRISP-DM write-up for the Data Preparation, EDA,
> Modeling, and Evaluation sections are already in the repo (`Data Prep - Derrick/`,
> `Analysis/`) as a starting point for the section owners to review and finalize.

---

## Scope & limitations

- We predict **observed** sales, not full unmet demand (`est_demand` is a rough, labelled proxy).
- **The 90-day window is the dataset's entire training history, not a sampling choice** —
  FreshRetailNet-50K ships exactly 90 train + 7 eval days for *every* series (even the full
  50,000-series dataset is only 90 days), so the data shows a weekly cycle but cannot show
  annual seasonality. A longer, multi-year dataset would be needed for seasonal forecasting.
- The **100-series** working set is a documented convenience sample; results should not be
  generalized to the full chain.

*Course project — shared for coursework and portfolio purposes.*
