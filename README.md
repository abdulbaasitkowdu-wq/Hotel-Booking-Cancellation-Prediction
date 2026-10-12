# Hotel Cancellation Risk Prediction

IT3091 Machine Learning (Y3.S1) — Group 26, Guided Data Track.

Predicts whether a hotel booking will cancel **before arrival**, then turns that probability into a **low / medium / high** risk tier a reservations manager can act on. The shipped model is **tuned histogram gradient boosting**.

This is **not** guest segmentation. There are no labelled segments to score.

## Problem

| | |
|---|---|
| **User** | Hotel reservations / revenue manager |
| **One row** | One booking |
| **Target** | `is_canceled` (binary) |
| **Output** | Cancel probability + risk tier (cutoffs 0.30 / 0.70, frozen on validation) |
| **Data** | 119,390 City and Resort bookings, 2015–2017 (~37% cancelled) |

Source: Antonio, Almeida & Nunes hotel booking demand datasethttps://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand.

## Method

1. **Time-aware split** by constructed `arrival_date` quantiles — not a shuffled split. Train until 10 Feb 2017, validation until 22 May 2017, test after that. The hotel will score *future* arrivals.
2. **Leakage-safe features.** Dropped `reservation_status`, `reservation_status_date`, and `assigned_room_type`. Impute, rare-category grouping, ADR clip, scaler, and one-hot encoder are **fit on train only**.
3. **Duplicates kept.** Extra copies look like group bookings, not data-entry errors.
4. **Four model families** on the same pipeline: logistic regression, depth-capped decision tree, random forest, histogram gradient boosting. `class_weight="balanced"`. Primary metric is **ROC-AUC**.
5. **Tuning** with Optuna TPE (20 trials) inside `TimeSeriesSplit(n_splits=3)` on **train only**. Forest tuning did not lift validation; boosting did.
6. **Test scored once.** Tiers and the shipped model were locked on validation first.

## Results (test set, n = 17,699)

Tuned `HistGradientBoostingClassifier`:

| Metric | Value |
|---|---|
| Accuracy | 0.790 |
| ROC-AUC | **0.882** |
| PR-AUC | 0.827 |
| Cancel precision / recall | 0.719 / 0.771 |
| Brier | 0.141 |

Risk tiers on test (actual cancel rate):

| Tier | Share of bookings | Actual cancel rate |
|---|---|---|
| Low (`p < 0.30`) | 7,014 | **6.5%** |
| Medium | 5,464 | 43.3% |
| High (`p ≥ 0.70`) | 5,221 | **79.9%** |

**How to use it:** treat high-risk bookings as a prompt to reconfirm or not count on occupancy. Do **not** auto-penalise a guest. Limits: 2015–2017 data, one chain, Non Refund is almost deterministic, Online TA is harder, no measured revenue lift.

## Repository layout

```
.
├── EDA.ipynb                          # full pipeline (EDA → models → test → tiers)
├── hotel_bookings(Original Data).csv  # raw dataset (rename before running — see below)
├── processed_data/                    # optional exports after cleaning / encoding
├── ML_Final_Report_corrected.docx     # submit-ready group report
├── Decision Log.docx
├── Personal_Learning_Journey_IT24103356.docx
└── README.md
```

Other `ML_Final_Report*.docx` files are drafts. Use **`ML_Final_Report_corrected.docx`**.

## How to run

1. Python 3.10+ with Jupyter.
2. Install:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn optuna
```

3. The notebook reads **`hotel_bookings.csv`**. This folder stores the same file as `hotel_bookings(Original Data).csv`. Copy or rename it first:

```bash
copy "hotel_bookings(Original Data).csv" hotel_bookings.csv
```

On macOS / Linux:

```bash
cp "hotel_bookings(Original Data).csv" hotel_bookings.csv
```

4. Open `EDA.ipynb` and run all cells **top to bottom**. Optuna tuning takes several minutes. Do not re-score the test set while changing models.

`processed_data/` is a convenience dump (human-readable processed frames, then scaled + one-hot frames). You do not need it to reproduce the notebook; the notebook rebuilds the pipeline from the raw CSV.

## Design choices worth knowing

- Split is chronological. Shuffled k-fold would leak later seasons into training.
- Accuracy is the wrong headline metric here; the product is a **ranked probability**.
- Logistic regression can look better on 0.5-threshold cancel recall. Boosting wins on ROC-AUC / Brier, which is what the tiers need.
- Calibration was **inspected**, not fitted as a post-hoc calibrator.

## Course

SLIIT IT3091 Machine Learning, 2026, Group 26 (Y3.S1.WD.AI.0101_G26).
