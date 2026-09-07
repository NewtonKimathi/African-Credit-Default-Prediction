"""
African Credit Scoring Challenge — Loan Default Prediction
=============================================================
Run:  python credit_default_model.py
Requires: pandas, numpy, scikit-learn, lightgbm  (pip install lightgbm)

The script is split into blocks (each starting with "# %%" so it also
opens as separate cells in VS Code / Jupyter). Each block has a one-line
title comment describing what it does.
"""

# %% ==========================================================
# BLOCK 0 — IMPORTS & CONFIG
# ==============================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
N_SPLITS = 5

DATA_DIR = "."          # change if your CSVs live elsewhere
TRAIN_PATH = f"{DATA_DIR}/Train.csv"
TEST_PATH = f"{DATA_DIR}/Test.csv"
ECON_PATH = f"{DATA_DIR}/economic_indicators.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/SampleSubmission.csv"


# %% ==========================================================
# BLOCK 1 — LOAD DATA
# ==============================================================
print("Loading data...")
train = pd.read_csv("C:\\Users\\pc\\Downloads\\Train.csv")
test = pd.read_csv("C:\\Users\\pc\\Downloads\\Test.csv")
econ = pd.read_csv("C:\\Users\\pc\\Downloads\\economic_indicators.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

train["is_train"] = 1
test["is_train"] = 0
test["target"] = np.nan

full = pd.concat([train, test], axis=0, ignore_index=True)


# %% ==========================================================
# BLOCK 2A — FEATURE ENGINEERING: DATES
# ==============================================================
full["disbursement_date"] = pd.to_datetime(full["disbursement_date"])
full["due_date"] = pd.to_datetime(full["due_date"])

full["disb_year"] = full["disbursement_date"].dt.year
full["disb_month"] = full["disbursement_date"].dt.month
full["disb_dayofweek"] = full["disbursement_date"].dt.dayofweek
full["disb_day"] = full["disbursement_date"].dt.day
full["due_dayofweek"] = full["due_date"].dt.dayofweek

# actual gap between disbursement and due date (should roughly equal duration
# but discrepancies can be informative, e.g. data entry issues / rescheduling)
full["date_gap_days"] = (full["due_date"] - full["disbursement_date"]).dt.days
full["gap_vs_duration_diff"] = full["date_gap_days"] - full["duration"]


# %% ==========================================================
# BLOCK 2B — FEATURE ENGINEERING: LOAN ECONOMICS RATIOS
# ==============================================================
eps = 1e-6
full["interest_amount"] = full["Total_Amount_to_Repay"] - full["Total_Amount"]
full["interest_rate"] = full["interest_amount"] / (full["Total_Amount"] + eps)
full["repay_to_principal_ratio"] = full["Total_Amount_to_Repay"] / (full["Total_Amount"] + eps)
full["daily_interest_rate"] = full["interest_rate"] / (full["duration"] + eps)
full["daily_repayment_amount"] = full["Total_Amount_to_Repay"] / (full["duration"] + eps)
full["amount_per_day"] = full["Total_Amount"] / (full["duration"] + eps)

full["lender_funded_gap"] = full["Total_Amount"] - full["Amount_Funded_By_Lender"]
full["lender_funded_ratio_actual"] = full["Amount_Funded_By_Lender"] / (full["Total_Amount"] + eps)
full["lender_repay_ratio"] = full["Lender_portion_to_be_repaid"] / (full["Amount_Funded_By_Lender"] + eps)
full["lender_portion_diff"] = full["Lender_portion_Funded"] - full["lender_funded_ratio_actual"]


# %% ==========================================================
# BLOCK 2C — FEATURE ENGINEERING: CATEGORICAL FLAGS & FREQUENCY ENCODING
# ==============================================================
full["is_new_loan"] = (full["New_versus_Repeat"] == "New Loan").astype(int)

for col in ["loan_type", "lender_id", "country_id"]:
    freq = full[col].value_counts(normalize=True)
    full[f"{col}_freq"] = full[col].map(freq)


# %% ==========================================================
# BLOCK 2D — FEATURE ENGINEERING: CUSTOMER HISTORY (leakage-safe, backward-looking only)
# ==============================================================
full = full.sort_values("disbursement_date").reset_index(drop=True)

full["cust_loan_seq"] = full.groupby("customer_id").cumcount()  # 0 = first loan seen
full["cust_prior_loans"] = full["cust_loan_seq"]

full["cust_prior_avg_amount"] = (
    full.groupby("customer_id")["Total_Amount"]
    .apply(lambda s: s.shift().expanding().mean())
    .reset_index(level=0, drop=True)
)
full["cust_prior_avg_duration"] = (
    full.groupby("customer_id")["duration"]
    .apply(lambda s: s.shift().expanding().mean())
    .reset_index(level=0, drop=True)
)
full["cust_prior_max_amount"] = (
    full.groupby("customer_id")["Total_Amount"]
    .apply(lambda s: s.shift().expanding().max())
    .reset_index(level=0, drop=True)
)

# Days since this customer's previous loan disbursement
full["cust_prev_disb_date"] = full.groupby("customer_id")["disbursement_date"].shift()
full["days_since_prev_loan"] = (
    full["disbursement_date"] - full["cust_prev_disb_date"]
).dt.days
full.drop(columns=["cust_prev_disb_date"], inplace=True)

# Customer's historical default rate (prior loans only -> no leakage)
train_mask = full["is_train"] == 1
full["cust_prior_default_rate"] = np.nan
prior_default_rate = (
    full.loc[train_mask]
    .groupby("customer_id")["target"]
    .apply(lambda s: s.shift().expanding().mean())
)
full.loc[train_mask, "cust_prior_default_rate"] = prior_default_rate.reset_index(level=0, drop=True).values

# For test rows (and train rows with no prior history) fill with global default rate
global_rate = train["target"].mean()
full["cust_prior_default_rate"] = full["cust_prior_default_rate"].fillna(global_rate)

# Recent-window default rate (last 20 known-outcome loans, not full history) —
# catches behavior drift that an all-time average smooths over.
recent_default_rate = (
    full.loc[train_mask]
    .groupby("customer_id")["target"]
    .apply(lambda s: s.shift().rolling(20, min_periods=1).mean())
)
full["cust_recent_default_rate"] = np.nan
full.loc[train_mask, "cust_recent_default_rate"] = recent_default_rate.reset_index(level=0, drop=True).values
full["cust_recent_default_rate"] = full["cust_recent_default_rate"].fillna(global_rate)

# Relative loan size: how large is this loan vs. what this customer normally borrows?
full["amount_vs_cust_avg"] = full["Total_Amount"] / (full["cust_prior_avg_amount"] + eps)
full["amount_vs_cust_avg"] = full["amount_vs_cust_avg"].fillna(1.0)  # no history -> neutral ratio

# Disbursement recency: continuous days-since-reference-date. NOTE: on this dataset
# this feature partly acts as a cleaner proxy for loan_type/product-mix shifts over
# time (91% of Train sits in a narrow Jul-Nov 2022 window dominated by one loan type),
# not purely a "risk trending over time" signal. Verified test shares the same
# date/product-mix pattern, so it transfers correctly -- but treat it as a snapshot
# artifact of this dataset, not a durable generalizable trend.
ref_date = full["disbursement_date"].min()
full["disb_days_since_ref"] = (full["disbursement_date"] - ref_date).dt.days


# %% ==========================================================
# BLOCK 2E — FEATURE ENGINEERING: LENDER & LOAN-TYPE HISTORICAL STATS (train-only, mapped onto full)
# ==============================================================
lender_stats = train.groupby("lender_id")["target"].agg(["mean", "count"]).rename(
    columns={"mean": "lender_hist_default_rate", "count": "lender_hist_loan_count"}
)
full = full.merge(lender_stats, on="lender_id", how="left")
full["lender_hist_default_rate"] = full["lender_hist_default_rate"].fillna(global_rate)
full["lender_hist_loan_count"] = full["lender_hist_loan_count"].fillna(0)

# Lender recent-window default rate (rolling last 500 known-outcome loans, not
# all-time average). Investigation found lender 267278's own default rate drops
# from 9.6% (pre-Jul-2022) to 0.9% (Jul-Nov 2022) -- a genuine shift in that
# lender's book over time, not just a loan-type mix effect. This feature lets
# the model see that drift directly instead of relying on disb_days_since_ref
# as an indirect proxy.
lender_recent_default_rate = (
    full.loc[train_mask]
    .groupby("lender_id")["target"]
    .apply(lambda s: s.shift().rolling(500, min_periods=1).mean())
)
full["lender_recent_default_rate"] = np.nan
full.loc[train_mask, "lender_recent_default_rate"] = lender_recent_default_rate.reset_index(level=0, drop=True).values

# As-of merge: for TEST rows (no target available), pull the lender's most
# recently known rolling rate from train as of that row's disbursement date,
# rather than a flat global fallback.
lender_asof = (
    full.loc[train_mask, ["lender_id", "disbursement_date", "lender_recent_default_rate"]]
    .dropna()
    .sort_values("disbursement_date")
)
full = full.sort_values("disbursement_date")
full = pd.merge_asof(
    full,
    lender_asof.rename(columns={"lender_recent_default_rate": "lender_recent_default_rate_asof"}),
    on="disbursement_date",
    by="lender_id",
    direction="backward",
)
full["lender_recent_default_rate"] = full["lender_recent_default_rate"].fillna(
    full["lender_recent_default_rate_asof"]
)
full.drop(columns=["lender_recent_default_rate_asof"], inplace=True)
full["lender_recent_default_rate"] = full["lender_recent_default_rate"].fillna(global_rate)

loantype_stats = train.groupby("loan_type")["target"].agg(["mean", "count"]).rename(
    columns={"mean": "loantype_hist_default_rate", "count": "loantype_hist_loan_count"}
)
full = full.merge(loantype_stats, on="loan_type", how="left")
full["loantype_hist_default_rate"] = full["loantype_hist_default_rate"].fillna(global_rate)
full["loantype_hist_loan_count"] = full["loantype_hist_loan_count"].fillna(0)


# %% ==========================================================
# BLOCK 2F — FEATURE ENGINEERING: MACRO-ECONOMIC INDICATORS (merged on country + disbursement year)
# ==============================================================
econ_long = econ.melt(id_vars=["Country", "Indicator"], var_name="Year", value_name="Value")
econ_long["Year"] = econ_long["Year"].str.replace("YR", "", regex=False).astype(int)
econ_wide = econ_long.pivot_table(index=["Country", "Year"], columns="Indicator", values="Value").reset_index()
econ_wide.columns = [
    c if c in ("Country", "Year") else "econ_" + c.split(",")[0].strip().lower().replace(" ", "_")
    for c in econ_wide.columns
]

full = full.merge(
    econ_wide, left_on=["country_id", "disb_year"], right_on=["Country", "Year"], how="left"
)
full.drop(columns=["Country", "Year"], inplace=True, errors="ignore")

econ_cols = [c for c in full.columns if c.startswith("econ_")]
for c in econ_cols:
    full[c] = full[c].fillna(full[c].median())

print(f"Total engineered feature count: {full.shape[1]}")

# Final categorical typing (label encode remaining string cols)
cat_cols = ["country_id", "loan_type", "New_versus_Repeat"]
for col in cat_cols:
    full[col] = full[col].astype("category")


# %% ==========================================================
# BLOCK 3 — BUILD FEATURE MATRIX (train/test split, X/y)
# ==============================================================
drop_cols = [
    "ID", "customer_id", "tbl_loan_id", "lender_id",
    "disbursement_date", "due_date", "target", "is_train",
]
feature_cols = [c for c in full.columns if c not in drop_cols]

train_df = full[full["is_train"] == 1].copy()
test_df = full[full["is_train"] == 0].copy()

X = train_df[feature_cols]
y = train_df["target"].astype(int)
X_test = test_df[feature_cols]

print(f"Final feature count: {len(feature_cols)}")


# %% ==========================================================
# BLOCK 4 — CROSS-VALIDATED TRAINING (LightGBM, imbalance-aware)
# ==============================================================
print("\nRunning 5-fold stratified cross-validation...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"scale_pos_weight (class imbalance ratio): {scale_pos_weight:.2f}")

lgb_params = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    fold_f1 = f1_score(y_val, (val_pred >= 0.5).astype(int))
    print(f"  Fold {fold}: F1@0.5 = {fold_f1:.4f}, best_iter = {model.best_iteration_}")


# %% ==========================================================
# BLOCK 5 — THRESHOLD TUNING (optimize F1 on out-of-fold predictions)
# ==============================================================
print("\nTuning classification threshold on OOF predictions...")

thresholds = np.arange(0.05, 0.95, 0.01)
f1_scores = [f1_score(y, (oof_preds >= t).astype(int)) for t in thresholds]
best_idx = int(np.argmax(f1_scores))
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best threshold: {best_threshold:.2f}  ->  OOF F1: {best_f1:.4f}")
print(f"F1 @ default 0.5 threshold: {f1_score(y, (oof_preds >= 0.5).astype(int)):.4f}")


# %% ==========================================================
# BLOCK 6 — FEATURE IMPORTANCE (from last fold model, illustrative)
# ==============================================================
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop 15 features by importance:")
print(importance.head(15).to_string(index=False))


# %% ==========================================================
# BLOCK 7 — FINAL PREDICTIONS -> SUBMISSION FILE
# ==============================================================
print("\nBuilding submission file...")

test_df["pred_proba"] = test_preds
test_df["target"] = (test_df["pred_proba"] >= best_threshold).astype(int)

submission = test_df[["ID", "target"]].rename(columns={"target": "Target"})
submission.to_csv("submission.csv", index=False)
print(f"Saved submission.csv  (predicted default rate: {submission['Target'].mean():.4f})")


# %% ==========================================================
# BLOCK 8 — BONUS: CREDIT SCORING FUNCTION (probability -> FICO-style score + risk band)
# ==============================================================
# Convert a default-probability into a 300–850 style credit score (like FICO):
# higher score = lower risk. We use a log-odds (logit) transform so the score
# scales linearly with odds-of-default, a standard scorecard technique.

def probability_to_score(p, base_score=600, base_odds=50, pdo=20):
    """
    Convert probability of default `p` into a scaled credit score.

    base_score: score assigned when odds-of-good/bad = base_odds
    base_odds : good:bad odds (e.g., 50 means 50 good loans per 1 default) at base_score
    pdo       : "points to double the odds" - how many score points it takes
                to double the good:bad odds (standard scorecard convention)
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = (1 - p) / p  # good:bad odds
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    score = offset + factor * np.log(odds)
    return np.round(score).astype(int)


def score_to_risk_band(score):
    if score >= 750:
        return "A - Very Low Risk"
    elif score >= 680:
        return "B - Low Risk"
    elif score >= 620:
        return "C - Medium Risk"
    elif score >= 550:
        return "D - High Risk"
    else:
        return "E - Very High Risk"


test_df["credit_score"] = probability_to_score(test_df["pred_proba"])
test_df["risk_band"] = test_df["credit_score"].apply(score_to_risk_band)

credit_scores = test_df[["ID", "pred_proba", "credit_score", "risk_band"]].rename(
    columns={"pred_proba": "default_probability"}
)
credit_scores.to_csv("credit_scores.csv", index=False)

print("\nSaved credit_scores.csv with columns: ID, default_probability, credit_score, risk_band")
print(credit_scores["risk_band"].value_counts())
print("\nDone.")
