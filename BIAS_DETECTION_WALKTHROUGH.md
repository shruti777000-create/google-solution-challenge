# AI Bias Detection Dashboard - Project Walkthrough

We have successfully built a full-stack AI Bias Detection application using Python and Anvil. This system allows users to upload datasets, train machine learning models, and automatically detect predictive bias across demographic groups.

## 🚀 Key Features Implemented

### 1. Interactive Frontend (Anvil UI)
- **Automatic Connection**: The `client_code.py` is designed to automatically hook into your Anvil designer components (`file_loader`, `run_button`, `output_text`, `plot_1`).
- **Dynamic Feedback**: Displays "Running analysis..." status and uses pop-up alerts for user errors (like forgetting to upload a file).
- **Visualization Display**: Seamlessly renders Matplotlib graphs generated on the server.

### 2. Machine Learning & Bias Pipeline
- **Data Processing**: Automatically cleans missing values and handles categorical encoding using `pd.get_dummies()`.
- **Model Training**: Implements a Logistic Regression model to predict outcomes (defaulting to the Adult Income dataset target).
- **Fairness Metrics**:
    - Calculates **Approval Rates** for Male vs Female groups.
    - Computes the **Bias Gap** (absolute difference).
    - **Automated Status**: Detects bias if the gap exceeds **20% (0.2)**.

### 3. Local-Cloud Integration (Anvil Uplink)
- **Bypass Cloud Limits**: Integrated Anvil Uplink so the heavy machine learning code runs on your local machine, avoiding Anvil "image build failed" errors.
- **Real-time Logging**: Added terminal logs (`🔄 Received file`, `🧠 Training model`, `✅ Analysis complete`) so you can monitor the backend locally.

## 🛠️ Technical Fixes Applied
- **AttributeErrors**: Resolved naming conflicts between the Anvil Designer and Python-generated properties.
- **Import Fixes**: Corrected the `anvil.BlobMedia` path for local Uplink compatibility.
- **Robustness**: Added fallbacks to detect gender columns even if they are named slightly differently in various datasets.

## 📊 How the System Works
1. **User** uploads a CSV.
2. **Frontend** sends the data to the **Local Server** via Uplink.
3. **Backend** trains the model and calculates the gap between Male and Female predictions.
4. **Backend** generates a color-coded bar chart and a text summary.
5. **Frontend** displays the results, showing if the model is **"✅ Fair"** or has **"⚠️ Bias Detected"**.
