import os
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from flask import Flask, request, jsonify, send_from_directory
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_folder='.')

# ──────────────────────────────────────────────
# AI BIAS LOGIC (Bulletproof Version)
# ──────────────────────────────────────────────

def clean_data(df):
    return df.dropna()

def train_model(X, y):
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        return model, predictions, accuracy
    except Exception as e:
        print(f"Model error: {e}")
        return None, None, 0

def detect_sensitive_column(X):
    """
    Tries to find a sensitive column. 
    Never returns None.
    """
    # 1. Look for common gender keywords
    for col in X.columns:
        c_low = col.lower()
        if 'sex_male' in c_low or 'gender_male' in c_low:
            return (X[col] == 1), (X[col] == 0), col
        if c_low == 'sex' or c_low == 'gender':
            vals = X[col].unique()
            if len(vals) >= 2:
                return (X[col] == vals[0]), (X[col] == vals[1]), col

    # 2. Look for ANY encoded column prefix
    generic_cols = [c for c in X.columns if '_' in c]
    if generic_cols:
        col = generic_cols[0]
        return (X[col] == 1), (X[col] == 0), col

    # 3. Last Resort: Just take the first column
    col = X.columns[0]
    vals = X[col].unique()
    if len(vals) >= 2:
        return (X[col] == vals[0]), (X[col] == vals[1]), col
    else:
        # If it's continuous, split by mean
        mean_val = X[col].mean()
        return (X[col] > mean_val), (X[col] <= mean_val), col

def calculate_fairness(predictions, is_male, is_female):
    if predictions is None: return 0, 0, 0
    male_rate   = float(predictions[is_male].mean()) if is_male.sum() > 0 else 0.0
    female_rate = float(predictions[is_female].mean()) if is_female.sum() > 0 else 0.0
    bias_gap    = abs(male_rate - female_rate)
    return male_rate, female_rate, bias_gap

def balance_dataset(X, y, is_male, is_female):
    X_male = X[is_male];     y_male = y[is_male]
    X_female = X[is_female];  y_female = y[is_female]
    
    # If one group is empty, we can't balance, just return original
    if len(X_male) == 0 or len(X_female) == 0:
        return X, y, is_male, is_female

    min_size = min(len(X_male), len(X_female))
    X_male_s = X_male.sample(n=min_size, random_state=42)
    y_male_s = y_male.loc[X_male_s.index]
    X_female_s = X_female.sample(n=min_size, random_state=42)
    y_female_s = y_female.loc[X_female_s.index]
    
    X_bal = pd.concat([X_male_s, X_female_s])
    y_bal = pd.concat([y_male_s, y_female_s])
    is_male_bal = pd.Series([True]*min_size + [False]*min_size, index=X_bal.index)
    is_female_bal = pd.Series([False]*min_size + [True]*min_size, index=X_bal.index)
    return X_bal, y_bal, is_male_bal, is_female_bal

# ──────────────────────────────────────────────
# WEB ROUTES
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        df = pd.read_csv(file)
        df = clean_data(df)

        # 1. Target Column (Smart & Safe)
        target_col = 'income_>50K'
        if target_col not in df.columns:
            target_col = df.columns[-1]
        
        y_raw = df[target_col]
        # Auto-encode target if it is text
        if y_raw.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
        else:
            y = y_raw

        # 2. Features (One-hot encode objects)
        X = pd.get_dummies(df.drop(columns=[target_col]))

        # 3. Sensitive Column (Bulletproof Detection)
        is_male, is_female, detected_name = detect_sensitive_column(X)

        # Step 1: Train Original
        _, predictions, accuracy = train_model(X, y)
        if predictions is None:
            return jsonify({"error": "This dataset is not compatible with Logistic Regression."}), 400

        male_before, female_before, bias_before = calculate_fairness(predictions, is_male, is_female)

        # Step 2: Mitigate & Retrain
        X_bal, y_bal, is_male_bal, is_female_bal = balance_dataset(X, y, is_male, is_female)
        _, predictions_bal, _ = train_model(X_bal, y_bal)
        male_after, female_after, bias_after = calculate_fairness(predictions_bal, is_male_bal, is_female_bal)

        # Step 3: Determine Status
        BIAS_THRESHOLD = 0.10
        status = "⚠️ Bias Detected" if bias_before > BIAS_THRESHOLD else "✅ Fair Model"
        
        improvement = bias_before - bias_after
        improvement_pct = (improvement / bias_before * 100) if bias_before > 0 else 0

        # Build custom labels based on what we detected
        group_a = "Group A"
        group_b = "Group B"
        if 'sex' in detected_name.lower() or 'gender' in detected_name.lower():
            group_a = "Male"
            group_b = "Female"

        return jsonify({
            "accuracy": round(accuracy, 4),
            "maleBefore": round(male_before, 4),
            "femaleBefore": round(female_before, 4),
            "biasBefore": round(bias_before, 4),
            "maleAfter": round(male_after, 4),
            "femaleAfter": round(female_after, 4),
            "biasAfter": round(bias_after, 4),
            "improvementPct": round(improvement_pct, 1),
            "status": status,
            "threshold": BIAS_THRESHOLD,
            "detectedAttribute": detected_name,
            "groupA": group_a,
            "groupB": group_b
        })

    except Exception as e:
        return jsonify({"error": f"Internal Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
