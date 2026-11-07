# MIS 480-50 — Retail Sales Patterns and Prediction

Predicting next-day retail sales with the **FreshRetailNet-50K** dataset — a group project
for **MIS 480-50 Predictive Analytics**, Metropolitan State University, Fall 2026.

**Team:** Paul Ries (project manager) · Feifei Li · Sem Masgina · Derrick Nyagesuka

**Headline result (official held-out week):** every model beats a previous-week baseline;
advance information gives a modest but real improvement, and a simple model (linear
regression or a shallow `log1p` Random Forest) generalizes best.

| Model | MAE | RMSE |
|---|---|---|
| baseline (previous-week same-day) | 0.601 | 0.960 |
| linear regression | 0.488 | **0.748** |
| random forest (`log1p`) | 0.477 | 0.796 |

## 📄 Documentation

- **[README](Analysis/Feifei%20Notebooks/README.md)** — overview, dataset, how to run, results, team roles
- **[CRISP-DM_Summary.md](Analysis/Feifei%20Notebooks/CRISP-DM_Summary.md)** — one-page project summary (also as `.docx`)

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
├── requirements.txt
├── Proposal/                           # project proposal
├── Requirements & Planning/            # team planning docs
├── Meeting Notes/                      # meeting notes
├── Final Report and Poster/            # final deliverables (in progress)
├── Data/                               # dataset landing (raw parquet git-ignored; see above)
│   ├── FreshRetailNet.html             # dataset card
│   └── Dataset Shortlist.docx          # dataset selection notes
├── Data Prep - Derrick/                # data preparation & ETL (Derrick)
│   ├── scripts/                        # 01_inspect · 02_subset · 03_eda (.py)
│   ├── data/subset/                    # derived 100-series working subset (CSV + parquet)
│   └── outputs/                        # data_dictionary, quality_report, charts (eda/, models/)
└── Analysis/
    ├── Feifei Notebooks/               # CRISP-DM notebooks + write-up (Feifei)
    │   ├── 01_inspect.ipynb            # data understanding
    │   ├── 02_subset.ipynb             # data preparation (100-series subset)
    │   ├── 03_eda.ipynb                # exploratory data analysis
    │   ├── 04_modeling_and_evaluation.ipynb   # modeling + evaluation (canonical)
    │   ├── 05_classification.ipynb     # classification sub-analysis
    │   ├── *.html                      # pre-rendered read-only exports
    │   ├── CRISP-DM_Summary.md / .docx # one-page summary
    │   └── README.md
    └── Sam Masgina/                    # modeling & evaluation (Sem)
        ├── MIS480_Modeling_Sem.ipynb   # next-day forecast: baseline / linear / Random Forest
        ├── Results_Summary.md          # test results + key findings
        └── fig_*.png                   # charts for the poster
```

**Notebooks run in order** (in `Analysis/Feifei Notebooks/`):
**01** inspect → **02** subset → **03** EDA → **04** modeling & evaluation → **05** classification.

---

## How to run

1. **Environment** (Python 3; conda or venv):
   ```bash
   pip install -r requirements.txt
   ```
2. **Get the data:** the working subset is already in `Data Prep - Derrick/data/subset/`.
   To rebuild it from scratch, download the dataset (see **Dataset** above) and run
   notebooks `01` → `02`.
3. **Run the notebooks** from the folder that contains the `data/` and `outputs/` paths
   (`Data Prep - Derrick/`), or open the pre-rendered `.html` exports for a quick read.

---

## Results at a glance

- **EDA:** sales rise with promotions/discounts/holidays; weather is weak; strong **weekly
  seasonality** (weekends ~50% above weekdays); stockout censoring confirmed.
- **Modeling:** next-day forecast with no-leakage features (past-sales lags + planned
  promo/holiday/discount + lagged weather/stockout); baseline vs. linear vs. Random Forest;
  `log1p` target transform; `max_depth` tuning; **95% prediction intervals** (~95% coverage).
- **Evaluation:** validated on the dataset's **official eval split** (true next 7 days).
- **Classification:** a "high- vs low-demand day" demo — logistic regression and a
  Random Forest classifier reach ~0.79 accuracy, ROC-AUC ~0.87.

Charts: `Data Prep - Derrick/outputs/eda/` and `.../outputs/models/`.

---

## Modeling & evaluation — Sem Masgina

Sem owns the **Modeling** and **Evaluation** sections. His notebook
[`Analysis/Sam Masgina/MIS480_Modeling_Sem.ipynb`](Analysis/Sam%20Masgina/MIS480_Modeling_Sem.ipynb)
builds the next-day forecast end to end (see his
[`Results_Summary.md`](Analysis/Sam%20Masgina/Results_Summary.md)):

- **No-leakage predictors** — every feature uses only information known the day before:
  planned `discount` / promotion / holiday / day-of-week, past-sales lags (1-day, 7-day,
  7-day average), and **yesterday's** stockout hours and weather.
- **Time-ordered split (2024)** — train Apr 4–May 28, validate May 29–Jun 11, test Jun 12–25.
  The Random Forest is tuned on the validation period, retrained on train+validation, then
  scored **once** on the untouched test period.
- **Metrics** — MAE, RMSE, and WAPE, with long-stockout days (12+ hours) checked separately.

**Test results** (test period Jun 12–25; lower is better):

| Model | MAE | RMSE | WAPE |
|---|---|---|---|
| baseline (same day last week) | 0.545 | 0.844 | 45.1% |
| linear regression | 0.455 | 0.671 | 37.6% |
| **Random Forest** | **0.438** | **0.663** | **36.3%** |

**Key findings:** Random Forest beats the baseline by ~20% (linear by ~17%); **past sales do
most of the work** (a past-sales-only forest scores MAE 0.439 vs 0.438); **holidays and
discounts** are the most useful factors beyond past sales, while previous-day weather and
stockout hours add almost nothing; excluding the 107 long-stockout test rows improves
Random Forest WAPE to **32.7%**.

> The team's canonical notebook `04_modeling_and_evaluation.ipynb` **adopts Sem's
> time-ordered split and tuned Random Forest**, and extends them with the dataset's official
> eval-week test, prediction intervals, permutation importance, and a short forecasting view.

---

## Project sections & ownership

**Paul Ries** acted as **project manager** — breaking the work into the sections below and
assigning owners (per the team's planning email):

| Section | Owner | Status |
|---|---|---|
| Dataset Overview | Feifei | ✅ Complete |
| Data Preparation | Derrick | ✅ Complete |
| EDA Results | Derrick / Feifei | ✅ Complete |
| Modeling | Sem | ✅ Notebook in repo |
| Evaluation Metrics | Feifei  | ✅ Notebook in repo |
| Presentation Slides | All | ⬜ Not started |
| Final Report | All | ⬜ Not started |

> Draft notebooks, charts, and the CRISP-DM write-up for the Data Preparation, EDA,
> Modeling, and Evaluation sections are already in the repo (`Data Prep - Derrick/`,
> `Analysis/`) as a starting point for the section owners to review and finalize.

---

## Scope & limitations

- We predict **observed** sales, not full unmet demand (`est_demand` is a rough, labelled proxy).
- A **90-day** window shows weekly but not annual seasonality.
- The **100-series** working set is a documented convenience sample; results should not be
  generalized to the full chain.

*Course project — shared for coursework and portfolio purposes.*
