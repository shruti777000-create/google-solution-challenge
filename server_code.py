import anvil.server
# --- UPLINK SETUP ---
# 1. Get your Uplink key from Anvil: Settings (Gear icon) -> Uplink -> Enable
# 2. Paste it inside the quotes below:
uplink_key = "server_AO22ERD47RQBKGYWCF5SAMZF-HUL7EQICRZ6N7XY2"

if uplink_key != "PASTE_YOUR_UPLINK_KEY_HERE":
    anvil.server.connect(uplink_key)
else:
    print("⚠️ WARNING: You need to paste your Uplink key into server_code.py!")
# --------------------
import anvil.media
import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@anvil.server.callable
def process_pipeline(file):
    print(f"🔄 Received file: {file.name}. Starting pipeline...")
    # Step 1: Load data
    if file is None:
        return {"error": "No file uploaded. Please upload a dataset."}
        
    try:
        # Read CSV using pandas
        with anvil.media.TempFile(file) as file_name:
            df = pd.read_csv(file_name)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return {"error": f"Error loading CSV file: {str(e)}"}
        
    # Step 2: Clean data
    df = df.dropna()
    
    # Step 4: Define target column
    target_col = 'income_>50K'
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found.")
        return {"error": f"Error: Required target column '{target_col}' not found in the dataset."}
        
    y = df[target_col]
    X_raw = df.drop(columns=[target_col])
    
    # Step 3: Encode categorical columns
    X = pd.get_dummies(X_raw)
    
    # Step 5: Train model
    print("🧠 Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Step 6: Predict
    predictions = model.predict(X)
    
    # Step 7: Calculate accuracy
    accuracy = accuracy_score(y, predictions)
    
    # Step 8: Fairness analysis
    if 'sex_Male' in X.columns:
        is_male = (X['sex_Male'] == 1)
        is_female = (X['sex_Male'] == 0)
    else:
        male_cols = [col for col in X.columns if 'sex' in col.lower() and 'male' in col.lower() and 'female' not in col.lower()]
        if male_cols:
            is_male = (X[male_cols[0]] == 1)
            is_female = (X[male_cols[0]] == 0)
        else:
            print("❌ Could not identify gender column.")
            return {"error": "Error: Could not identify 'sex_Male' column after encoding."}
            
    male_approval_rate = predictions[is_male].mean() if is_male.sum() > 0 else 0
    female_approval_rate = predictions[is_female].mean() if is_female.sum() > 0 else 0
    
    bias_gap = abs(male_approval_rate - female_approval_rate)
    
    # Step 9: Determine bias
    if bias_gap > 0.2:
        bias_status = "⚠️ Bias Detected"
    else:
        bias_status = "✅ Fair Model"
        
    # Step 10: Create visualization
    print("📊 Generating plot...")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Male', 'Female'], [male_approval_rate, female_approval_rate], color=['#4facfe', '#fa709a'])
    ax.set_ylabel('Approval Rate')
    ax.set_title('Approval Rates by Gender')
    ax.set_ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    # Using anvil.BlobMedia instead of anvil.media.BlobMedia for Uplink compatibility
    plot_media = anvil.BlobMedia('image/png', buf.read(), name='fairness_plot.png')
    plt.close(fig)
    
    summary_text = (
        f"Model Accuracy: {accuracy:.0%}\n"
        f"Male: {male_approval_rate:.0%}\n"
        f"Female: {female_approval_rate:.0%}\n"
        f"Bias Gap: {bias_gap:.0%}\n"
        f"Status: {bias_status}"
    )
    
    print("✅ Analysis complete!")
    return {
        "summary": summary_text,
        "plot": plot_media
    }

if __name__ == "__main__":
    if uplink_key != "PASTE_YOUR_UPLINK_KEY_HERE":
        print("🚀 Server is running locally and connected to Anvil! Waiting for clicks...")
        anvil.server.wait_forever()
