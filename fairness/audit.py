# audit.py
# Person 3 — Fairness Engineer
# Day 3: Run the full audit on biased model vs fixed model
# and produce a comparison result for Person 4's dashboard

import pandas as pd
import numpy as np
import json
import os
import sys

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from fairness import run_full_audit   # your functions from Day 2


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BIASED_PRED_PATH = "../model/predictions.csv"        # original biased model
FIXED_PRED_PATH  = "../model/predictions_fixed.csv"  # retrained fair model
OUTPUT_PATH      = "audit_results.json"              # what P4 loads
SENSITIVE_COLS   = ["sex", "race"]                   # groups to audit (changed gender -> sex)
PRED_COL         = "predicted"
ACTUAL_COL       = "actual"


# ─────────────────────────────────────────────
# STEP 1 — LOAD PREDICTIONS
# ─────────────────────────────────────────────

def load_predictions(path, label):
    """
    Loads a predictions CSV and validates required columns exist.
    """
    print(f"\n📂 Loading {label} predictions: {path}")

    if not os.path.exists(path):
        print(f"   ⚠️  File not found: {path}")
        return None

    df = pd.read_csv(path)
    print(f"   ✅ Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")

    # Validate required columns
    required = [PRED_COL, ACTUAL_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"   ❌ Missing columns: {missing}")
        return None

    return df


# ─────────────────────────────────────────────
# STEP 2 — AUDIT ONE MODEL
# ─────────────────────────────────────────────

def audit_model(df, model_label):
    """
    Runs all 3 fairness metrics across all sensitive columns
    for one model's predictions.

    Returns a structured dict of all results.
    """
    print(f"\n{'='*50}")
    print(f"  AUDITING: {model_label}")
    print(f"{'='*50}")

    model_results = {
        "model_label": model_label,
        "total_rows":  len(df),
        "audits":      {}
    }

    for col in SENSITIVE_COLS:
        if col not in df.columns:
            print(f"\n⚠️  Column '{col}' not found — skipping")
            continue

        print(f"\n🔍 Auditing by: {col}")
        result = run_full_audit(df, PRED_COL, ACTUAL_COL, col)
        model_results["audits"][col] = result

    return model_results


# ─────────────────────────────────────────────
# STEP 3 — COMPARE BIASED VS FIXED
# ─────────────────────────────────────────────

def compare_models(biased_results, fixed_results):
    """
    Compares fairness scores between the biased and fixed model.
    Shows exactly how much each metric improved.

    This is the "proof the fix worked" section.
    """
    print(f"\n{'='*50}")
    print(f"  COMPARISON: Biased vs Fixed Model")
    print(f"{'='*50}")

    comparison = {}

    for col in SENSITIVE_COLS:

        biased_audit = biased_results["audits"].get(col)
        fixed_audit  = fixed_results["audits"].get(col) if fixed_results else None

        if not biased_audit:
            continue

        col_comparison = {}

        metrics = [
            ("demographic_parity", "gap"),
            ("equal_opportunity",  "gap"),
            ("fpr_parity",         "gap"),
        ]

        print(f"\n📋 Group: {col}")
        print(f"   {'Metric':<25} {'Biased':>10} {'Fixed':>10} {'Improvement':>12} {'Status':>8}")
        print(f"   {'-'*65}")

        for metric_key, gap_key in metrics:
            biased_gap = biased_audit.get(metric_key, {}).get(gap_key)
            fixed_gap  = fixed_audit.get(metric_key, {}).get(gap_key) if fixed_audit else None

            if biased_gap is None:
                continue

            biased_str = f"{biased_gap:.4f}"
            fixed_str  = f"{fixed_gap:.4f}"  if fixed_gap  is not None else "pending"
            improvement = round(biased_gap - fixed_gap, 4) if fixed_gap is not None else None
            imp_str    = f"{improvement:+.4f}" if improvement is not None else "pending"

            # Did it improve?
            if improvement is not None:
                status = "✅ Better" if improvement > 0 else ("➡️ Same" if improvement == 0 else "❌ Worse")
            else:
                status = "⏳"

            print(f"   {metric_key:<25} {biased_str:>10} {fixed_str:>10} {imp_str:>12} {status:>8}")

            col_comparison[metric_key] = {
                "biased_gap":   biased_gap,
                "fixed_gap":    fixed_gap,
                "improvement":  improvement,
                "improved":     improvement > 0 if improvement is not None else None
            }

        comparison[col] = col_comparison

    return comparison


# ─────────────────────────────────────────────
# STEP 4 — SAVE RESULTS FOR PERSON 4
# ─────────────────────────────────────────────

def save_results(biased_results, fixed_results, comparison):
    """
    Saves everything to audit_results.json
    Person 4 loads this file directly into the dashboard.

    Structure:
    {
      "biased_model":  { all audit results },
      "fixed_model":   { all audit results },
      "comparison":    { before/after gaps },
      "summary":       { high level pass/fail counts }
    }
    """

    # Build a plain-English summary
    summary = build_summary(biased_results, fixed_results)

    output = {
        "biased_model": biased_results,
        "fixed_model":  fixed_results,
        "comparison":   comparison,
        "summary":      summary
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Audit results saved to: {OUTPUT_PATH}")
    print(f"   (Share this path with Person 4 for the dashboard)")


def build_summary(biased_results, fixed_results):
    """
    Builds a simple high-level summary card.
    This is what goes in the big headline on the dashboard.
    """
    summary = {
        "biased_model": {},
        "fixed_model":  {}
    }

    for label, results in [("biased_model", biased_results),
                            ("fixed_model",  fixed_results)]:
        if not results:
            summary[label] = {"status": "not available"}
            continue

        total_audits = 0
        passed       = 0

        for col, audit in results["audits"].items():
            for metric in ["demographic_parity", "equal_opportunity", "fpr_parity"]:
                m = audit.get(metric, {})
                if "passed" in m:
                    total_audits += 1
                    if m["passed"]:
                        passed += 1

        failed = total_audits - passed
        summary[label] = {
            "total_checks": total_audits,
            "passed":       passed,
            "failed":       failed,
            "verdict":      "FAIR" if failed == 0 else "BIASED",
            "score":        f"{passed}/{total_audits} checks passed"
        }

    return summary


# ─────────────────────────────────────────────
# MAIN — runs the full Day 3 audit
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 50)
    print("  BIAS DETECTION PROJECT — Day 3 Audit")
    print("  Person 3 — Fairness Engineer")
    print("=" * 50)

    # Load predictions
    df_biased = load_predictions(BIASED_PRED_PATH, "Biased Model")
    df_fixed  = load_predictions(FIXED_PRED_PATH,  "Fixed Model")

    # If neither file exists — create demo data to prove the pipeline works
    if df_biased is None:
        print("\n⚠️  No predictions found — generating demo data")
        print("   Replace with real files from Person 2 when ready\n")

        # Biased model — heavily favours men
        df_biased = pd.DataFrame({
            "actual":    [1]*50 + [0]*50 + [1]*50 + [0]*50,
            "predicted": [1]*40 + [0]*10 + [0]*50 +   # women: low TPR
                         [1]*45 + [0]*5  + [1]*20 + [0]*30,  # men: high TPR
            "sex":       ["Female"]*100 + ["Male"]*100,
            "race":      (["White"]*25 + ["Black"]*25 + ["Asian"]*25 + ["Other"]*25) * 2
        })

        # Fixed model — more balanced
        df_fixed = pd.DataFrame({
            "actual":    [1]*50 + [0]*50 + [1]*50 + [0]*50,
            "predicted": [1]*44 + [0]*6  + [0]*50 +   # women: improved TPR
                         [1]*46 + [0]*4  + [1]*12 + [0]*38,  # men: reduced FPR
            "sex":       ["Female"]*100 + ["Male"]*100,
            "race":      (["White"]*25 + ["Black"]*25 + ["Asian"]*25 + ["Other"]*25) * 2
        })

    # Audit both models
    biased_results = audit_model(df_biased, "Original Biased Model")
    fixed_results  = audit_model(df_fixed,  "Fixed Model") if df_fixed is not None else None

    # Compare them
    comparison = compare_models(biased_results, fixed_results)

    # Save for Person 4
    save_results(biased_results, fixed_results, comparison)

    print("\n✅ Day 3 complete!")
    print("   → audit_results.json ready for Person 4")
    print("   → Push to GitHub and notify Person 4")
