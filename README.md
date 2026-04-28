# AI Bias Detection Dashboard — Upgraded Setup Guide

> This system helps ensure fair and responsible AI decisions.

This guide walks you through setting up the **upgraded** AI Bias Detection Dashboard in Anvil.  
The app now supports: dataset upload → model training → bias detection → **bias mitigation** → before/after comparison.

---

## Step 1: Create / Open Your Anvil App

1. Go to [Anvil](https://anvil.works/) and log in.
2. Open your existing app **or** click **Create a New App** → choose **Material Design 3**.

---

## Step 2: Build the UI — Required Components

In the Anvil **Design** view for `Form1`, add the following components.  
**Component names must match exactly** (set them in the Properties panel on the right).

### 📂 Dataset Section
| Component | Name | Settings |
|---|---|---|
| Label | *(any)* | Text: `📂 Dataset`, Role: `Headline` |
| FileLoader | `file_loader` | Text: `Upload CSV Dataset`, FileTypes: `.csv` |
| Button | `run_button` | Text: `▶ Run Analysis`, Role: `Primary Color` |

### 📊 Model Results Section
| Component | Name | Settings |
|---|---|---|
| Label | *(any)* | Text: `📊 Model Results`, Role: `Headline` |
| Label | `output_text` | Text: *(blank)*, Allow whitespace: ✅ |

### ⚖️ Bias Analysis Section
| Component | Name | Settings |
|---|---|---|
| Label | *(any)* | Text: `⚖️ Bias Analysis`, Role: `Headline` |
| Label | `status_label` | Text: *(blank)*, Bold: ✅ |
| Image | `plot_1` | *(blank source)* — "Before Bias Mitigation" chart |

### 🛠 Bias Mitigation Section
| Component | Name | Settings |
|---|---|---|
| Label | *(any)* | Text: `🛠 Bias Mitigation`, Role: `Headline` |
| Image | `plot_after` | *(blank source)* — "After Bias Mitigation" chart |

### Optional
| Component | Name | Settings |
|---|---|---|
| Button | `download_button` | Text: `📄 Download Report` |
| Label | *(any)* | Text: `This system helps ensure fair and responsible AI decisions.` |

---

## Step 3: Add Client Code

1. In the Anvil editor, go to `Form1` → **Code** view.
2. Replace all code with the contents of **`client_code.py`** in this folder.

---

## Step 4: Add Server Code

> If you use **Anvil Uplink** (running `server_code.py` locally):
> - Ensure your Uplink key is pasted into `uplink_key` in `server_code.py`.
> - Run `python server_code.py` in your terminal.

> If you use an **Anvil Server Module** (hosted):
> - Copy `server_code.py` content into your Server Module.
> - Remove the `anvil.server.connect(...)` and `anvil.server.wait_forever()` lines.

---

## Step 5: Python Packages

Ensure these packages are available in your environment:

```
pandas
scikit-learn
matplotlib
```

*(Anvil hosted environments include these by default.)*

---

## Step 6: Run and Test

1. Click **Run** at the top of the Anvil editor.
2. Upload a CSV with:
   - A `sex` or `gender` column (the sensitive attribute)
   - An `income_>50K` column (the target label)
   - *(The Adult Census Income dataset works perfectly)*
3. Click **▶ Run Analysis** and wait for results.

---

## What the App Does

| Stage | Description |
|---|---|
| **Load & Clean** | Reads CSV, drops missing rows |
| **Encode** | One-hot encodes categorical columns |
| **Train** | Logistic Regression on original data |
| **Detect Bias** | Compares Male vs Female approval rates |
| **Mitigate** | Downsamples larger group → balanced dataset |
| **Retrain** | Logistic Regression on balanced data |
| **Compare** | Shows Before vs After bias gap |
| **Visualize** | Two bar charts side-by-side |

---

## Output Format

```
Model Accuracy: XX%

Before Bias:
  Male Approval:   XX%
  Female Approval: XX%
  Bias Gap:        XX%

After Bias Fix:
  Male Approval:   XX%
  Female Approval: XX%
  Bias Gap:        XX%

Final Status: ⚠️ Bias Detected  OR  ✅ Fair Model
```

**Status indicator:**
- 🔴 `⚠️ Bias Detected` — bias gap > 20%
- 🟢 `✅ Fair Model`   — bias gap ≤ 20%

---

## Code Architecture

| Function | Purpose |
|---|---|
| `clean_data(df)` | Drop NA rows |
| `train_model(X, y)` | Fit LogisticRegression, return predictions + accuracy |
| `detect_sensitive_column(X)` | Smart detection of sex/gender column |
| `calculate_fairness(...)` | Compute approval rates & bias gap |
| `balance_dataset(...)` | Downsample larger group for fairness |
| `create_plot(...)` | Styled bar chart → BlobMedia PNG |
| `process_pipeline(file)` | Orchestrates the full pipeline |
