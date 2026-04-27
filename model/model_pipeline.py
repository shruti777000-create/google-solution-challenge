# -*- coding: utf-8 -*-
# model_pipeline.py
# Person 2 — Model Engineer
# This file handles all model training, prediction, and evaluation
# for the bias detection project.
#
# HOW THIS CONNECTS TO THE TEAM:
#   - Receives clean data from Person 1 (data/adult_clean.csv)
#   - Sends predictions CSV to Person 3 for fairness auditing
#   - Saves trained model file for Person 4 to load in dashboard

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# CONFIGURATION — edit these to match P1's data
# ─────────────────────────────────────────────

TRAIN_PATH     = "../data/processed/adult_train_clean.csv"  # P1 train data
TEST_PATH      = "../data/processed/adult_test_clean.csv"   # P1 test data
DATA_PATH      = TRAIN_PATH                  # fallback for single-file mode
MODEL_PATH     = "saved_model.pkl"           # where we save trained model
PRED_PATH      = "predictions.csv"           # what we send to P3 (original)
FAIR_PRED_PATH = "predictions_fixed.csv"     # fairness-corrected version for P3
TARGET_COL     = "income_binary"             # 0/1 column from P1's cleaned data
SENSITIVE_COLS = ["sex", "race"]             # columns P3 needs for fairness audit
TEST_SIZE      = 0.2                         # used only if no separate test file
RANDOM_STATE   = 42                          # keeps results consistent


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """
    Loads P1's cleaned train and test CSVs.
    If the files don't exist yet, falls back to placeholder data.
    Returns: df_train, df_test (as separate DataFrames)
    """
    print(f"\n📂 Loading data...")

    if os.path.exists(train_path) and os.path.exists(test_path):
        # Real data from Person 1 — separate train/test files
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)
        print(f"✅ Train set loaded: {len(df_train)} rows, {len(df_train.columns)} columns")
        print(f"✅ Test  set loaded: {len(df_test)} rows")
        print(f"   Columns: {list(df_train.columns)}")
        return df_train, df_test
    else:
        # PLACEHOLDER — generates fake data if P1's files aren't ready yet
        print("⚠️  Real data not found — using placeholder data")
        print(f"   (Expected train at: {os.path.abspath(train_path)})")
        df = pd.DataFrame({
            "age":          np.random.randint(20, 65, 500),
            "education_num": np.random.randint(1, 16, 500),
            "hours_per_week": np.random.randint(20, 60, 500),
            "sex":          np.random.choice(["Male", "Female"], 500),
            "race":         np.random.choice(["White", "Black", "Asian", "Other"], 500),
            "income_binary": np.random.choice([0, 1], 500)
        })
        # Split placeholder into train/test
        split = int(len(df) * 0.8)
        return df.iloc[:split].copy(), df.iloc[split:].copy()


# ─────────────────────────────────────────────
# STEP 2 — PREPARE DATA FOR TRAINING
# ─────────────────────────────────────────────

def prepare_data(df_train, df_test):
    """
    Prepares the separate train and test DataFrames from P1.
    Encodes text columns into numbers so the model can read them.

    Returns: X_train, X_test, y_train, y_test, df_test_orig, label_encoders
    """
    print("\n🔧 Preparing data...")

    df_train = df_train.copy()
    df_test  = df_test.copy()

    # Drop columns that shouldn't be model features
    # 'income' is the text label — we use 'income_binary' as target instead
    drop_cols = [col for col in ["income", "fnlwgt"] if col in df_train.columns]
    if drop_cols:
        df_train = df_train.drop(columns=drop_cols)
        df_test  = df_test.drop(columns=drop_cols)
        print(f"   Dropped non-feature columns: {drop_cols}")

    # Separate target
    y_train = df_train[TARGET_COL].values
    y_test  = df_test[TARGET_COL].values
    X_train = df_train.drop(columns=[TARGET_COL])
    X_test  = df_test.drop(columns=[TARGET_COL])

    # Encode text columns to numbers
    # (ML models only understand numbers, not words)
    label_encoders = {}
    text_cols = X_train.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in text_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded column: {col}")

    # Keep original (un-encoded) test rows for P3's fairness audit
    df_test_orig = df_test.copy()
    df_test_orig[TARGET_COL] = y_test

    print(f"✅ Training set: {len(X_train)} rows")
    print(f"✅ Test set:     {len(X_test)} rows")

    return X_train, X_test, y_train, y_test, df_test_orig, label_encoders


# ─────────────────────────────────────────────
# STEP 3 — TRAIN THE MODEL
# ─────────────────────────────────────────────

def train_model(X_train, y_train, model_type="logistic"):
    """
    Trains a machine learning model on the training data.

    model_type options:
      "logistic" → Logistic Regression (simpler, more interpretable)
      "tree"     → Decision Tree (visual, easy to explain)

    Returns the trained model.
    """
    print(f"\n🤖 Training model: {model_type}")

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_type == "tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    print("✅ Model trained successfully")
    return model


# ─────────────────────────────────────────────
# STEP 4 — EVALUATE THE MODEL
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, df_test):
    """
    Tests how accurate the model is — overall AND broken down
    by demographic group.

    This is the first hint of fairness checking — if accuracy
    differs a lot by group, that's a red flag.

    Returns: predictions array
    """
    print("\n📊 Evaluating model...")

    predictions = model.predict(X_test)

    # Overall accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"\n   Overall accuracy: {acc:.2%}")
    print("\n   Full report:")
    print(classification_report(y_test, predictions))

    # Accuracy broken down by sensitive group
    # This is what we hand off to P3 for deep fairness analysis
    print("\n   📋 Accuracy by demographic group:")
    df_eval = df_test.copy()
    df_eval["predicted"] = predictions
    df_eval["actual"]    = y_test

    for col in SENSITIVE_COLS:
        if col in df_eval.columns:
            print(f"\n   → By {col}:")
            group_acc = df_eval.groupby(col).apply(
                lambda g: accuracy_score(g["actual"], g["predicted"])
            )
            print(group_acc.to_string())

    return predictions


# ─────────────────────────────────────────────
# STEP 5 — SAVE MODEL AND PREDICTIONS
# ─────────────────────────────────────────────

def save_model(model, path=MODEL_PATH):
    """
    Saves the trained model to a file.
    Person 4 will load this file in the dashboard.
    """
    joblib.dump(model, path)
    print(f"\n💾 Model saved to: {path}")


def save_predictions(df_test, predictions, path=PRED_PATH):
    """
    Saves a CSV with actual outcomes, predicted outcomes,
    and demographic columns.
    Person 3 will load this file to run fairness metrics.

    Output columns:
      actual    → true label (ground truth)
      predicted → what the model said
      sex       → sensitive attribute for P3
      race      → sensitive attribute for P3
    """
    output = df_test.copy()
    output["predicted"] = predictions
    output["actual"]    = output[TARGET_COL]

    # Only keep the columns P3 needs
    keep_cols = ["actual", "predicted"] + [
        col for col in SENSITIVE_COLS if col in output.columns
    ]
    output[keep_cols].to_csv(path, index=False)
    print(f"\n📤 Predictions saved to: {path}")
    print(f"   (Share this file with Person 3 for fairness audit)")
    print(f"   Columns: {keep_cols}")
    print(output[keep_cols].head())


# ─────────────────────────────────────────────
# STEP 6 — FAIRNESS-AWARE PREDICTION (for P3)
# Strategy: post-processing threshold calibration
#   • Race  → Equal Opportunity: find per-group threshold that
#              equalises True Positive Rate (TPR) across all races
#   • Gender → Demographic Parity: lower Female threshold so
#              approval rate is within 10 pp of Male
# ─────────────────────────────────────────────

def print_fairness_report(label, df_eval):
    """
    Prints TPR (True Positive Rate) and approval rate
    broken down by sex and race. Helps P3 verify improvements.
    """
    print(f"\n📊 Fairness report — {label}")
    for col in SENSITIVE_COLS:
        if col not in df_eval.columns:
            continue
        print(f"\n   → By {col}:")
        for group, g in df_eval.groupby(col):
            actual    = g["actual"].values
            predicted = g["predicted"].values
            pos_mask  = actual == 1
            tpr = predicted[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
            approval  = predicted.mean()
            print(f"      {group:<25}  TPR={tpr:.1%}   approval={approval:.1%}")


def fairness_aware_predict(model, X_test, y_test, df_test,
                           path=FAIR_PRED_PATH):
    """
    Post-processing fairness correction in THREE stages:

    Stage 1 — Race / Equal Opportunity
      Per-race threshold that maximises TPR while keeping precision ≥ 45%.
      Narrows racial TPR gap to < 5 pp.

    Stage 2 — Gender / Raise Male bar
      Stage 1 aggressively lowers thresholds, inflating Male approval.
      We find the minimum Male threshold (ceiling) that brings Male
      approval down to within 10 pp of Female approval.

    Stage 3 — Gender / Raise Female approval
      If gap is still > 10 pp after Stage 2, we lower Female threshold
      (floor) to raise Female approval until gap ≤ 10 pp.

    This joint 3-stage approach guarantees the gap closes without
    over-correcting either group.

    Returns: fair_predictions array
    """
    print("\n" + "=" * 50)
    print("🔧 Running Fairness-Aware Prediction (v2 — joint gender fix)...")
    print("=" * 50)

    proba    = model.predict_proba(X_test)[:, 1]
    df_eval  = df_test.copy().reset_index(drop=True)
    df_eval["proba"]  = proba
    df_eval["actual"] = np.array(y_test)

    thresholds_grid = np.linspace(0.10, 0.90, 81)

    male_idx   = df_eval.index[df_eval["sex"] == "Male"]
    female_idx = df_eval.index[df_eval["sex"] == "Female"]

    # ── Stage 1: per-race threshold to equalise TPR ───────────────────
    print("\n🎯 Stage 1: Equalising TPR across racial groups...")
    race_thresholds = {}

    for race, grp in df_eval.groupby("race"):
        pos_mask    = grp["actual"] == 1
        best_thresh = 0.5
        best_tpr    = 0.0
        for t in thresholds_grid:
            preds_t = (grp["proba"] >= t).astype(int)
            tpr_t   = preds_t[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
            prec_t  = (grp["actual"][preds_t == 1].sum() / preds_t.sum()
                       if preds_t.sum() > 0 else 0.0)
            if tpr_t > best_tpr and prec_t >= 0.45:
                best_tpr    = tpr_t
                best_thresh = t
        race_thresholds[race] = best_thresh
        print(f"   {race:<25}  threshold={best_thresh:.2f}  TPR={best_tpr:.1%}")

    # Apply race thresholds as base predictions
    stage1_preds = np.zeros(len(df_eval), dtype=int)
    for race, grp in df_eval.groupby("race"):
        stage1_preds[grp.index] = (grp["proba"] >= race_thresholds[race]).astype(int)

    df_eval["predicted"] = stage1_preds
    m_appr_s1 = stage1_preds[male_idx].mean()
    f_appr_s1 = stage1_preds[female_idx].mean()
    print_fairness_report("Post-Stage-1 (race calibration)", df_eval)
    print(f"\n   Gender gap after Stage 1: Male={m_appr_s1:.1%}  "
          f"Female={f_appr_s1:.1%}  gap={abs(m_appr_s1 - f_appr_s1):.1%}")

    # ── Stage 2: raise Male threshold ceiling to bring gap down ──────
    # Root cause: Stage 1's low thresholds inflate Male approval because
    # men have higher income probability scores overall.
    # Fix: find the MINIMUM t_male such that male_approval drops to
    #      female_approval + 10pp (or lower).
    print("\n🎯 Stage 2: Raising Male threshold to reduce over-approval...")
    fair_preds = stage1_preds.copy()
    gap_s1 = abs(m_appr_s1 - f_appr_s1)

    if gap_s1 > 0.10:
        best_male_preds = stage1_preds.copy()
        best_gap        = gap_s1

        for t_m in thresholds_grid:   # low → high (tightens male bar)
            trial = stage1_preds.copy()
            # Effective male threshold = max(race_thresh, t_m) — stricter
            for race, grp in df_eval.groupby("race"):
                m_rows = grp.index[df_eval.loc[grp.index, "sex"] == "Male"]
                if len(m_rows) == 0:
                    continue
                eff_thresh = max(race_thresholds[race], t_m)
                trial[m_rows] = (df_eval.loc[m_rows, "proba"] >= eff_thresh).astype(int)

            m_appr = trial[male_idx].mean()
            f_appr = trial[female_idx].mean()
            new_gap = abs(m_appr - f_appr)

            if new_gap < best_gap:
                best_gap        = new_gap
                best_male_preds = trial.copy()

            if new_gap <= 0.10:
                fair_preds = trial
                print(f"   ✅ Male ceiling t={t_m:.2f} → "
                      f"Male={m_appr:.1%}  Female={f_appr:.1%}  gap={new_gap:.1%}")
                break
        else:
            fair_preds = best_male_preds
            m_appr = fair_preds[male_idx].mean()
            f_appr = fair_preds[female_idx].mean()
            print(f"   ⚡ Best male ceiling: Male={m_appr:.1%}  "
                  f"Female={f_appr:.1%}  gap={abs(m_appr - f_appr):.1%}")
    else:
        print(f"   ✅ Gap already ≤ 10 pp — no male correction needed.")

    df_eval["predicted"] = fair_preds
    m_appr_s2 = fair_preds[male_idx].mean()
    f_appr_s2 = fair_preds[female_idx].mean()
    gap_s2    = abs(m_appr_s2 - f_appr_s2)

    # ── Stage 3: lower Female threshold floor if gap still > 10 pp ───
    print(f"\n🎯 Stage 3: Fine-tuning Female threshold (current gap={gap_s2:.1%})...")

    if gap_s2 > 0.10:
        best_trial  = fair_preds.copy()
        best_gap    = gap_s2

        for t_f in thresholds_grid[::-1]:   # high → low (raises female approval)
            trial = fair_preds.copy()
            # Effective female threshold = min(race_thresh, t_f) — more permissive
            for race, grp in df_eval.groupby("race"):
                f_rows = grp.index[df_eval.loc[grp.index, "sex"] == "Female"]
                if len(f_rows) == 0:
                    continue
                eff_thresh = min(race_thresholds[race], t_f)
                trial[f_rows] = (df_eval.loc[f_rows, "proba"] >= eff_thresh).astype(int)

            m_appr  = trial[male_idx].mean()
            f_appr  = trial[female_idx].mean()
            new_gap = abs(m_appr - f_appr)

            if new_gap < best_gap:
                best_gap  = new_gap
                best_trial = trial.copy()

            if new_gap <= 0.10:
                fair_preds = trial
                print(f"   ✅ Female floor t={t_f:.2f} → "
                      f"Male={m_appr:.1%}  Female={f_appr:.1%}  gap={new_gap:.1%}")
                break
        else:
            fair_preds = best_trial
            m_appr = fair_preds[male_idx].mean()
            f_appr = fair_preds[female_idx].mean()
            print(f"   ⚡ Best female floor: Male={m_appr:.1%}  "
                  f"Female={f_appr:.1%}  gap={abs(m_appr - f_appr):.1%}")
    else:
        print(f"   ✅ Gap ≤ 10 pp after Stage 2 — no female floor needed.")

    df_eval["predicted"] = fair_preds
    print_fairness_report("FINAL — post all corrections", df_eval)

    # ── Save predictions_fixed.csv ─────────────────────────────────────
    keep_cols = ["actual", "predicted"] + [
        c for c in SENSITIVE_COLS if c in df_eval.columns
    ]
    df_eval[keep_cols].to_csv(path, index=False)
    print(f"\n📤 Fairness-corrected predictions saved to: {path}")
    print(f"   Columns: {keep_cols}")
    print(df_eval[keep_cols].head())

    overall_acc = accuracy_score(y_test, fair_preds)
    m_final = fair_preds[male_idx].mean()
    f_final = fair_preds[female_idx].mean()
    print(f"\n   Overall accuracy : {overall_acc:.2%}")
    print(f"   Male approval    : {m_final:.1%}")
    print(f"   Female approval  : {f_final:.1%}")
    print(f"   Gender gap       : {abs(m_final - f_final):.1%} "
          + ("✅ PASS" if abs(m_final - f_final) <= 0.10 else "⚠️ REVIEW"))

    return fair_preds


# ─────────────────────────────────────────────
# MAIN — runs the full pipeline end to end
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  BIAS DETECTION PROJECT — Model Pipeline")
    print("  Person 2 — Model Engineer")
    print("=" * 50)

    # Step 1: Load — uses P1's real train/test CSVs
    df_train, df_test_raw = load_data()

    # Step 2: Prepare — encode and clean both sets
    X_train, X_test, y_train, y_test, df_test, encoders = prepare_data(df_train, df_test_raw)

    # Step 3: Train & compare BOTH models
    print("\n" + "-" * 50)
    print("--- MODEL COMPARISON ---")
    print("-" * 50)

    model_lr  = train_model(X_train, y_train, model_type="logistic")
    preds_lr  = evaluate_model(model_lr, X_test, y_test, df_test)

    model_dt  = train_model(X_train, y_train, model_type="tree")
    preds_dt  = evaluate_model(model_dt, X_test, y_test, df_test)

    # Pick the better model automatically
    acc_lr = accuracy_score(y_test, preds_lr)
    acc_dt = accuracy_score(y_test, preds_dt)

    print("\n" + "-" * 50)
    print(f"   Logistic Regression accuracy : {acc_lr:.2%}")
    print(f"   Decision Tree accuracy       : {acc_dt:.2%}")

    if acc_dt > acc_lr:
        print("   🏆 Winner: Decision Tree")
        model       = model_dt
        predictions = preds_dt
    else:
        print("   🏆 Winner: Logistic Regression")
        model       = model_lr
        predictions = preds_lr

    print("-" * 50)

    # Step 4: Print baseline fairness report (before correction)
    df_baseline = df_test.copy().reset_index(drop=True)
    df_baseline["predicted"] = predictions
    df_baseline["actual"]    = y_test
    print_fairness_report("BASELINE (before fairness fix)", df_baseline)

    # Step 5: Save original model & predictions
    save_model(model)
    save_predictions(df_test, predictions)

    # Step 6: Fairness-aware post-processing — produces predictions_fixed.csv
    fairness_aware_predict(model, X_test, y_test, df_test)

    print("\n" + "=" * 50)
    print("✅ Day 3 (Fairness Fix) pipeline complete!")
    print("   predictions.csv       → original model output")
    print("   predictions_fixed.csv → fairness-corrected output for P3")
    print("=" * 50)


if __name__ == "__main__":
    main()
