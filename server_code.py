import anvil.server
import anvil.media
import anvil
import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- UPLINK SETUP ---
# 1. Get your Uplink key from Anvil: Settings (Gear icon) -> Uplink -> Enable
# 2. Paste it inside the quotes below:
uplink_key = "server_AO22ERD47RQBKGYWCF5SAMZF-HUL7EQICRZ6N7XY2"

if uplink_key != "PASTE_YOUR_UPLINK_KEY_HERE":
    anvil.server.connect(uplink_key)
else:
    print("⚠️ WARNING: You need to paste your Uplink key into server_code.py!")
# --------------------


# ──────────────────────────────────────────────
# MODULE: clean_data
# ──────────────────────────────────────────────
def clean_data(df):
    """Drop rows with missing values and return cleaned DataFrame."""
    return df.dropna()


# ──────────────────────────────────────────────
# MODULE: train_model
# ──────────────────────────────────────────────
def train_model(X, y):
    """
    Train a Logistic Regression model.
    Returns (model, predictions, accuracy).
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return model, predictions, accuracy


# ──────────────────────────────────────────────
# MODULE: detect_sensitive_column
# ──────────────────────────────────────────────
def detect_sensitive_column(X):
    """
    Smart detection of the gender/sex sensitive column from encoded features.
    Handles: 'sex_Male', 'gender_Male', raw 'sex'/'gender' columns.
    Returns (is_male_mask, is_female_mask) as boolean pandas Series,
    or (None, None) if not found.
    """
    # Priority 1: exact encoded column 'sex_Male'
    if 'sex_Male' in X.columns:
        return (X['sex_Male'] == 1), (X['sex_Male'] == 0)

    # Priority 2: any col containing 'sex' + 'male' but not 'female'
    male_cols = [
        col for col in X.columns
        if 'sex' in col.lower() and 'male' in col.lower() and 'female' not in col.lower()
    ]
    if male_cols:
        return (X[male_cols[0]] == 1), (X[male_cols[0]] == 0)

    # Priority 3: gender_Male or similar
    gender_male_cols = [
        col for col in X.columns
        if 'gender' in col.lower() and 'male' in col.lower() and 'female' not in col.lower()
    ]
    if gender_male_cols:
        return (X[gender_male_cols[0]] == 1), (X[gender_male_cols[0]] == 0)

    # Priority 4: look for a female-encoded column and invert
    female_cols = [
        col for col in X.columns
        if ('sex' in col.lower() or 'gender' in col.lower()) and 'female' in col.lower()
    ]
    if female_cols:
        return (X[female_cols[0]] == 0), (X[female_cols[0]] == 1)

    # Not found
    return None, None


# ──────────────────────────────────────────────
# MODULE: calculate_fairness
# ──────────────────────────────────────────────
def calculate_fairness(predictions, is_male, is_female):
    """
    Compute male/female approval rates and the absolute bias gap.
    Returns (male_rate, female_rate, bias_gap).
    """
    male_rate   = float(predictions[is_male].mean())   if is_male.sum()   > 0 else 0.0
    female_rate = float(predictions[is_female].mean()) if is_female.sum() > 0 else 0.0
    bias_gap    = abs(male_rate - female_rate)
    return male_rate, female_rate, bias_gap


# ──────────────────────────────────────────────
# MODULE: balance_dataset
# ──────────────────────────────────────────────
def balance_dataset(X, y, is_male, is_female):
    """
    Downsample the larger gender group to match the smaller group size.
    Returns (X_balanced, y_balanced, is_male_bal, is_female_bal).
    """
    X_male   = X[is_male];   y_male   = y[is_male]
    X_female = X[is_female]; y_female = y[is_female]

    min_size = min(len(X_male), len(X_female))

    # Sample without replacement using fixed seed for reproducibility
    X_male_s   = X_male.sample(n=min_size,   random_state=42)
    y_male_s   = y_male.loc[X_male_s.index]
    X_female_s = X_female.sample(n=min_size, random_state=42)
    y_female_s = y_female.loc[X_female_s.index]

    X_bal = pd.concat([X_male_s, X_female_s])
    y_bal = pd.concat([y_male_s, y_female_s])

    # Rebuild boolean masks aligned to balanced index
    is_male_bal   = pd.Series([True]  * min_size + [False] * min_size, index=X_bal.index)
    is_female_bal = pd.Series([False] * min_size + [True]  * min_size, index=X_bal.index)

    return X_bal, y_bal, is_male_bal, is_female_bal


# ──────────────────────────────────────────────
# MODULE: create_plot
# ──────────────────────────────────────────────
def create_plot(male_rate, female_rate, title):
    """
    Create a styled bar chart comparing Male vs Female approval rates.
    Returns an anvil.BlobMedia PNG image.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#4facfe', '#fa709a']
    bars = ax.bar(['Male', 'Female'], [male_rate, female_rate],
                  color=colors, width=0.5, edgecolor='white', linewidth=1.2)

    ax.set_ylabel('Approval Rate', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.1%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6), textcoords='offset points',
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    media = anvil.BlobMedia('image/png', buf.read(), name=f'{title}.png')
    plt.close(fig)
    return media


# ──────────────────────────────────────────────
# MAIN CALLABLE: process_pipeline
# ──────────────────────────────────────────────
@anvil.server.callable
def process_pipeline(file):
    """
    Full pipeline:
      1. Load & clean CSV
      2. Encode features, train model
      3. Detect bias BEFORE mitigation
      4. Balance dataset & retrain
      5. Detect bias AFTER mitigation
      6. Generate two comparison charts
      7. Return structured result dict
    """
    print(f"🔄 Received file: {file.name}. Starting pipeline...")

    # ── Step 1: Load data ──
    if file is None:
        return {"error": "No file uploaded. Please upload a dataset."}
    try:
        with anvil.media.TempFile(file) as file_name:
            df = pd.read_csv(file_name)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return {"error": f"Error loading CSV file: {str(e)}"}

    # ── Step 2: Clean data ──
    df = clean_data(df)

    # ── Step 3: Identify target column (Smart Detection) ──
    potential_targets = ['income_>50K', 'target', 'label', 'outcome', 'default', 'survived', 'class']
    target_col = next((pt for pt in potential_targets if pt in df.columns), df.columns[-1])
    
    if len(df.columns) < 2:
        return {"error": "Dataset must have at least two columns (Features and Target)."}

    # Encode target if it's categorical (e.g., 'Yes'/'No' -> 1/0)
    y = df[target_col]
    if y.dtype == 'object' or y.nunique() == 2:
        y = pd.factorize(y)[0]
    
    X_raw = df.drop(columns=[target_col])
    if X_raw.empty:
        return {"error": "No features found to train the model."}

    # ── Step 4: Encode categorical features ──
    X = pd.get_dummies(X_raw)

    # ── Step 5: Detect sensitive column (Smart Detection) ──
    is_male, is_female = detect_sensitive_column(X)
    
    # Broad Search if specific gender detection fails
    if is_male is None:
        sensitive_keywords = ['sex', 'gender', 'race', 'age', 'ethnicity', 'nationality', 'origin']
        found_col = next((col for col in X_raw.columns if any(kw in col.lower() for kw in sensitive_keywords)), None)
        
        if found_col:
            vals = X_raw[found_col].unique()
            if len(vals) >= 2:
                is_male = (X_raw[found_col] == vals[0])
                is_female = (X_raw[found_col] == vals[1])
                print(f"ℹ️ Detected sensitive column '{found_col}'. Comparing '{vals[0]}' vs '{vals[1]}'.")

    if is_male is None or is_male.sum() == 0 or is_female.sum() == 0:
        return {"error": "Could not identify a sensitive attribute with at least two distinct groups (e.g., Male/Female or Race). Please check your column names."}

    # ── Step 6: Train original model ──
    print("🧠 Training original model...")
    _, predictions, accuracy = train_model(X, y)

    # ── Step 7: Fairness BEFORE mitigation ──
    male_before, female_before, bias_before = calculate_fairness(predictions, is_male, is_female)
    print(f"📐 Bias BEFORE: {bias_before:.2%}")

    # ── Step 8: Balance dataset ──
    print("⚖️ Applying bias mitigation (downsampling)...")
    X_bal, y_bal, is_male_bal, is_female_bal = balance_dataset(X, y, is_male, is_female)

    # ── Step 9: Retrain on balanced dataset ──
    print("🧠 Retraining on balanced dataset...")
    _, predictions_bal, accuracy_bal = train_model(X_bal, y_bal)

    # ── Step 10: Fairness AFTER mitigation ──
    male_after, female_after, bias_after = calculate_fairness(
        predictions_bal, is_male_bal, is_female_bal
    )
    print(f"📐 Bias AFTER:  {bias_after:.2%}")

    # ── Step 11: Determine final status ──
    # Threshold set to 0.1 (10%) — a 10%+ gap between groups signals meaningful bias
    BIAS_THRESHOLD = 0.10
    status = "⚠️ Bias Detected" if bias_before > BIAS_THRESHOLD else "✅ Fair Model"

    # ── Step 12: Generate comparison plots ──
    print("📊 Generating plots...")
    plot_before = create_plot(male_before, female_before, "Before Bias Mitigation")
    plot_after  = create_plot(male_after,  female_after,  "After Bias Mitigation")

    # ── Step 13: Build enhanced summary ──
    # Calculate how much bias improved after mitigation
    improvement = bias_before - bias_after
    improvement_pct = (improvement / bias_before * 100) if bias_before > 0 else 0

    summary_text = (
        f"Model Accuracy: {accuracy:.0%}\n\n"
        f"Before Bias:\n"
        f"  Male Approval:   {male_before:.0%}\n"
        f"  Female Approval: {female_before:.0%}\n"
        f"  Bias Gap:        {bias_before:.0%}\n\n"
        f"After Bias Fix:\n"
        f"  Male Approval:   {male_after:.0%}\n"
        f"  Female Approval: {female_after:.0%}\n"
        f"  Bias Gap:        {bias_after:.0%}\n"
        f"  Improvement:     {improvement_pct:.1f}% reduction in bias\n\n"
        f"Final Status: {status}"
    )

    print("✅ Analysis complete!")
    return {
        "summary":     summary_text,
        "plot_before": plot_before,
        "plot_after":  plot_after,
        "bias_before": bias_before,
        "bias_after":  bias_after,
        "threshold":   BIAS_THRESHOLD,
    }


if __name__ == "__main__":
    if uplink_key != "PASTE_YOUR_UPLINK_KEY_HERE":
        print("🚀 Server running locally and connected to Anvil! Waiting for clicks...")
        anvil.server.wait_forever()
