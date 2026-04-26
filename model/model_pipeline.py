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

DATA_PATH      = "../data/adult_clean.csv"   # P1 will put data here
MODEL_PATH     = "saved_model.pkl"           # where we save trained model
PRED_PATH      = "predictions.csv"           # what we send to P3
TARGET_COL     = "income"                    # column we're predicting
SENSITIVE_COLS = ["gender", "race"]          # columns P3 needs for fairness audit
TEST_SIZE      = 0.2                         # 20% of data used for testing
RANDOM_STATE   = 42                          # keeps results consistent


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────

def load_data(path=DATA_PATH):
    """
    Loads the clean CSV from Person 1.
    Returns a pandas DataFrame.
    """
    print(f"\n📂 Loading data from: {path}")

    if os.path.exists(path):
        # Real data from Person 1
        df = pd.read_csv(path)
        print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        return df
    else:
        # PLACEHOLDER — generates fake data if P1's file isn't ready yet
        print("⚠️  adult_clean.csv not found — using placeholder data")
        print(f"   (Expected at: {os.path.abspath(path)})")
        df = pd.DataFrame({
            "age":        np.random.randint(20, 65, 500),
            "education":  np.random.randint(1, 16, 500),
            "hours":      np.random.randint(20, 60, 500),
            "gender":     np.random.choice(["Male", "Female"], 500),
            "race":       np.random.choice(["White", "Black", "Asian", "Other"], 500),
            "income":     np.random.choice([0, 1], 500)   # 1 = high income
        })
        return df


# ─────────────────────────────────────────────
# STEP 2 — PREPARE DATA FOR TRAINING
# ─────────────────────────────────────────────

def prepare_data(df):
    """
    Splits data into features (X) and target (y).
    Encodes any text columns into numbers so the model can read them.
    Splits into training set and test set.

    Returns: X_train, X_test, y_train, y_test, df_test
    """
    print("\n🔧 Preparing data...")

    df = df.copy()

    # Separate target column
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Encode text columns to numbers
    # (ML models only understand numbers, not words like "Male"/"Female")
    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"   Encoded column: {col}")

    # Encode target if needed
    if y.dtype == object:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Train/test split
    # 80% of data trains the model, 20% is kept aside to test it
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Keep the original (un-encoded) test rows for P3's fairness audit
    df_test = df.iloc[X_test.index].copy()

    print(f"✅ Training set: {len(X_train)} rows")
    print(f"✅ Test set:     {len(X_test)} rows")

    return X_train, X_test, y_train, y_test, df_test, label_encoders


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
      gender    → sensitive attribute for P3
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
# MAIN — runs the full pipeline end to end
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  BIAS DETECTION PROJECT — Model Pipeline")
    print("  Person 2 — Model Engineer")
    print("=" * 50)

    # Step 1: Load
    df = load_data()

    # Step 2: Prepare
    X_train, X_test, y_train, y_test, df_test, encoders = prepare_data(df)

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

    # Step 5: Save the winner
    save_model(model)
    save_predictions(df_test, predictions)

    print("\n" + "=" * 50)
    print("✅ Day 2 pipeline complete!")
    print("   Models compared, best one saved.")
    print("=" * 50)


if __name__ == "__main__":
    main()
