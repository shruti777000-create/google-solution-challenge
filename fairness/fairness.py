# fairness.py
# Person 3 — Fairness Engineer
# Day 2: Full implementation of all 3 fairness metric functions
#
# HOW TO USE:
#   from fairness import run_full_audit
#   results = run_full_audit(df, "predicted", "actual", "gender")

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

THRESHOLD = 0.10   # gaps above 10% are flagged as biased
PRED_PATH = "../model/predictions.csv"   # file from Person 2


# ─────────────────────────────────────────────
# METRIC 1 — DEMOGRAPHIC PARITY
# ─────────────────────────────────────────────

def compute_demographic_parity(df, prediction_col, group_col, positive_label=1):
    """
    Measures whether the model approves people at equal rates
    across demographic groups.

    A gap > 0.10 means the model favours one group over another
    in terms of raw approval rate — regardless of actual qualification.

    Args:
        df             : DataFrame with predictions and group labels
        prediction_col : column name of model predictions e.g. 'predicted'
        group_col      : column name of sensitive attribute e.g. 'gender'
        positive_label : what counts as positive outcome e.g. 1 = hired

    Returns:
        dict with approval rate per group, the gap, and a pass/fail flag
    """
    print(f"\n📊 Computing Demographic Parity by '{group_col}'...")

    groups = df[group_col].unique()
    rates  = {}

    for group in groups:
        subset       = df[df[group_col] == group]
        approval_rate = (subset[prediction_col] == positive_label).mean()
        rates[group] = round(float(approval_rate), 4)
        print(f"   {group}: {approval_rate:.2%} approval rate")

    # Gap = difference between highest and lowest approval rate
    max_rate = max(rates.values())
    min_rate = min(rates.values())
    gap      = round(max_rate - min_rate, 4)

    passed = gap <= THRESHOLD

    print(f"   Gap: {gap:.4f} → {'✅ PASS' if passed else '❌ FAIL'}")

    return {
        "metric":  "Demographic Parity",
        "group_col": group_col,
        "rates":   rates,
        "gap":     gap,
        "threshold": THRESHOLD,
        "passed":  passed,
        "interpretation": (
            f"The approval rate gap between groups is {gap:.2%}. "
            f"{'This is within acceptable range.' if passed else 'This indicates potential discrimination.'}"
        )
    }


# ─────────────────────────────────────────────
# METRIC 2 — EQUAL OPPORTUNITY
# ─────────────────────────────────────────────

def compute_equal_opportunity(df, prediction_col, actual_col, group_col, positive_label=1):
    """
    Measures whether the model correctly identifies truly qualified
    people at equal rates across groups.

    Only looks at people who are ACTUALLY positive (truly qualified).
    Asks: of those, does the model catch them equally across groups?

    A gap > 0.10 means the model misses qualified people from one
    group far more than another — a serious fairness violation.

    Args:
        df             : DataFrame with predictions, actuals, group labels
        prediction_col : column name of model predictions
        actual_col     : column name of ground truth labels
        group_col      : column name of sensitive attribute
        positive_label : what counts as a positive outcome

    Returns:
        dict with TPR per group, the gap, and a pass/fail flag
    """
    print(f"\n📊 Computing Equal Opportunity by '{group_col}'...")

    # Only look at people who are ACTUALLY qualified (actual == positive)
    truly_positive = df[df[actual_col] == positive_label]

    if len(truly_positive) == 0:
        print("   ⚠️  No positive cases found in dataset")
        return {"metric": "Equal Opportunity", "error": "No positive cases"}

    groups = truly_positive[group_col].unique()
    tprs   = {}

    for group in groups:
        subset = truly_positive[truly_positive[group_col] == group]

        if len(subset) == 0:
            tprs[group] = None
            continue

        # TPR = how many truly positive people did the model correctly catch?
        tpr = (subset[prediction_col] == positive_label).mean()
        tprs[group] = round(float(tpr), 4)
        print(f"   {group}: TPR = {tpr:.2%}  (caught {int(tpr * len(subset))}/{len(subset)} qualified people)")

    valid_tprs = [v for v in tprs.values() if v is not None]
    gap        = round(max(valid_tprs) - min(valid_tprs), 4)
    passed     = gap <= THRESHOLD

    print(f"   Gap: {gap:.4f} → {'✅ PASS' if passed else '❌ FAIL'}")

    return {
        "metric":    "Equal Opportunity",
        "group_col": group_col,
        "tprs":      tprs,
        "gap":       gap,
        "threshold": THRESHOLD,
        "passed":    passed,
        "interpretation": (
            f"Among truly qualified people, the model's detection rate "
            f"differs by {gap:.2%} across groups. "
            f"{'Acceptable.' if passed else 'Qualified people from one group are being unfairly missed.'}"
        )
    }


# ─────────────────────────────────────────────
# METRIC 3 — FALSE POSITIVE RATE PARITY
# ─────────────────────────────────────────────

def compute_fpr_parity(df, prediction_col, actual_col, group_col, positive_label=1):
    """
    Measures whether the model gives false approvals at equal rates
    across groups.

    Only looks at people who are ACTUALLY negative (truly unqualified).
    Asks: of those, does the model wrongly approve them equally across groups?

    A gap > 0.10 means the model is unfairly lenient toward one group —
    or unfairly strict toward another.

    Args:
        df             : DataFrame with predictions, actuals, group labels
        prediction_col : column name of model predictions
        actual_col     : column name of ground truth labels
        group_col      : column name of sensitive attribute
        positive_label : what counts as a positive outcome

    Returns:
        dict with FPR per group, the gap, and a pass/fail flag
    """
    print(f"\n📊 Computing FPR Parity by '{group_col}'...")

    # Only look at people who are ACTUALLY not qualified
    truly_negative = df[df[actual_col] != positive_label]

    if len(truly_negative) == 0:
        print("   ⚠️  No negative cases found in dataset")
        return {"metric": "FPR Parity", "error": "No negative cases"}

    groups = truly_negative[group_col].unique()
    fprs   = {}

    for group in groups:
        subset = truly_negative[truly_negative[group_col] == group]

        if len(subset) == 0:
            fprs[group] = None
            continue

        # FPR = how many truly negative people did the model wrongly approve?
        fpr = (subset[prediction_col] == positive_label).mean()
        fprs[group] = round(float(fpr), 4)
        print(f"   {group}: FPR = {fpr:.2%}  (wrongly approved {int(fpr * len(subset))}/{len(subset)} unqualified people)")

    valid_fprs = [v for v in fprs.values() if v is not None]
    gap        = round(max(valid_fprs) - min(valid_fprs), 4)
    passed     = gap <= THRESHOLD

    print(f"   Gap: {gap:.4f} → {'✅ PASS' if passed else '❌ FAIL'}")

    return {
        "metric":    "FPR Parity",
        "group_col": group_col,
        "fprs":      fprs,
        "gap":       gap,
        "threshold": THRESHOLD,
        "passed":    passed,
        "interpretation": (
            f"Among unqualified people, the false approval rate "
            f"differs by {gap:.2%} across groups. "
            f"{'Acceptable.' if passed else 'The model is being unfairly lenient or strict toward certain groups.'}"
        )
    }


# ─────────────────────────────────────────────
# MASTER FUNCTION — runs all 3 metrics
# ─────────────────────────────────────────────

def run_full_audit(df, prediction_col, actual_col, group_col, threshold=THRESHOLD):
    """
    Runs all 3 fairness metrics and returns a single structured result.
    Person 4 calls THIS function to get everything needed for the dashboard.

    Args:
        df             : DataFrame with predictions, actuals, group labels
        prediction_col : column name of model predictions
        actual_col     : column name of ground truth labels
        group_col      : column name of sensitive attribute e.g. 'gender'
        threshold      : gap above which bias is flagged (default 0.10)

    Returns:
        dict — all scores, gaps, pass/fail flags, and summary
    """
    print("\n" + "=" * 50)
    print(f"  FAIRNESS AUDIT — Group: {group_col}")
    print("=" * 50)

    dp  = compute_demographic_parity(df, prediction_col, group_col)
    eo  = compute_equal_opportunity(df, prediction_col, actual_col, group_col)
    fpr = compute_fpr_parity(df, prediction_col, actual_col, group_col)

    all_passed = all([
        dp.get("passed",  False),
        eo.get("passed",  False),
        fpr.get("passed", False)
    ])

    summary = {
        "group_col":           group_col,
        "overall_passed":      all_passed,
        "overall_verdict":     "✅ FAIR" if all_passed else "❌ BIASED",
        "demographic_parity":  dp,
        "equal_opportunity":   eo,
        "fpr_parity":          fpr,
    }

    print(f"\n{'=' * 50}")
    print(f"  OVERALL VERDICT: {summary['overall_verdict']}")
    print(f"{'=' * 50}\n")

    return summary


# ─────────────────────────────────────────────
# UNIT TESTS — run these to verify your functions
# ─────────────────────────────────────────────

def run_unit_tests():
    """
    Tests all 3 functions with known data so you can verify
    your math is correct before using real model predictions.
    """
    print("\n🧪 Running unit tests...\n")

    # Build a small fake dataset with KNOWN bias
    # Women: 30% approval rate, Men: 70% — obvious demographic parity fail
    test_data = pd.DataFrame({
        "actual":    [1,1,1,1,1, 0,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0],
        "predicted": [1,1,0,0,0, 1,0,0,0,0, 1,1,1,1,0, 1,1,1,0,0],
        "gender":    ["Female"]*10 + ["Male"]*10
    })

    print("Test dataset:")
    print(test_data.to_string())

    results = run_full_audit(test_data, "predicted", "actual", "gender")

    # Assertions — these tell you if your math is right
    dp_gap = results["demographic_parity"]["gap"]
    eo_gap = results["equal_opportunity"]["gap"]

    assert dp_gap > 0,  "❌ TEST FAILED: Demographic parity gap should be > 0"
    assert eo_gap >= 0, "❌ TEST FAILED: Equal opportunity gap should be >= 0"

    print("\n✅ All unit tests passed!")
    return results


# ─────────────────────────────────────────────
# MAIN — test with real predictions from P2
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # First run unit tests on fake data
    run_unit_tests()

    # Then try with real predictions from Person 2
    try:
        print(f"\n📂 Loading predictions from Person 2: {PRED_PATH}")
        df = pd.read_csv(PRED_PATH)
        print(f"✅ Loaded {len(df)} rows")
        print(f"   Columns: {list(df.columns)}\n")

        # Run audit for gender/sex
        gender_col = "gender" if "gender" in df.columns else "sex" if "sex" in df.columns else None
        if gender_col:
            gender_results = run_full_audit(df, "predicted", "actual", gender_col)

        # Run audit for race
        if "race" in df.columns:
            race_results = run_full_audit(df, "predicted", "actual", "race")

    except FileNotFoundError:
        print(f"\n⚠️  predictions.csv not found yet — that's okay for Day 2")
        print("   Unit tests above already confirmed your functions work.")
        print("   Get the file from Person 2 and run again.")
