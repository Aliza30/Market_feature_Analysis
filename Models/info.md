# 📊 Marketing Campaign Analytics & Prediction

A two-part data science project that analyzes marketing campaign performance and builds machine learning models to predict ROI outcomes.

---

## 📁 Project Structure

```
├── marketing_optimization.ipynb   # EDA & ROI analysis across channels, duration, and audience
├── prediction_models.ipynb        # ML models to predict ROI and campaign success
├── updated_marketing_campaign_dataset.csv
└── models/
    └── roi_model.pkl              # Saved Random Forest model
```

---

## 📋 Dataset

**File:** `updated_marketing_campaign_dataset.csv`

Each row represents one marketing campaign with the following fields:

| Column | Type | Description |
|---|---|---|
| `Campaign_ID` | int | Unique campaign identifier |
| `Company` | str | Company running the campaign |
| `Campaign_Type` | str | Email, Display, Influencer, or Search |
| `Duration` | str | Duration as string (e.g. "30 days") |
| `Duration_Days` | int | Duration as numeric (days) |
| `Channel_Used` | str | Facebook, YouTube, Google Ads, Instagram, Email, Website |
| `Target_Gender` | str | Men, Women, or All |
| `Target_Age` | str | Age group (e.g. 18–24, 25–34) |
| `ROI` | float | Return on Investment |
| `Acquisition_Cost` | str | Cost to acquire a customer (stored as "$X,XXX.XX") |
| `Conversion_Rate` | float | Percentage of users who converted |
| `Clicks` | int | Number of clicks |
| `Impressions` | int | Number of impressions |
| `Engagement_Score` | int | Engagement score (1–10) |
| `Customer_Segment` | str | Audience segment (e.g. Health & Wellness, Fashionistas) |
| `Location` | str | City where the campaign ran |
| `Language` | str | Language of the campaign |
| `Date` | str | Campaign date |

---

## 📓 Notebook 1 — `marketing_optimization.ipynb`

**Goal:** Answer key business questions through exploratory data analysis and visualizations.

### Business Questions Answered

| Question | Finding |
|---|---|
| Best campaign duration? | **Below 30 days** (ROI ≈ 5.00) |
| Best marketing channel? | **Facebook** (ROI ≈ 5.02) |
| Best target audience? | **All genders** (ROI ≈ 5.01) |
| Lowest acquisition cost? | **YouTube** (avg. $12,481) |
| Best overall combination? | **Display + Website + All** (ROI ≈ 5.11) |

### Analysis Steps

1. **Load & Preview Data** — import libraries, load CSV, inspect first 5 rows
2. **Feature Engineering** — bin `Duration_Days` into categories: `Below 30`, `30-60`, `60+`
3. **Duration ROI Analysis** — groupby duration category → bar chart
4. **Channel ROI Analysis** — groupby channel → sorted horizontal bar chart
5. **Audience ROI Analysis** — groupby target gender → bar chart
6. **Acquisition Cost Cleaning** — strip `$` and `,` from string values → convert to float
7. **Cost Analysis** — groupby channel → sorted horizontal bar chart (zoomed x-axis)
8. **Multi-Variable Grouping** — groupby Campaign Type + Channel + Gender → top 10 combos
9. **ROI Heatmap** — pivot table visualized as a `seaborn` heatmap

### Libraries

```python
pandas        # Data manipulation
matplotlib    # Charts and plots
seaborn       # Heatmaps
```

---

## 📓 Notebook 2 — `prediction_models.ipynb`

**Goal:** Build machine learning models to predict ROI (regression) and campaign success (classification).

### Models Built

| Model | Task | Target | Result |
|---|---|---|---|
| Random Forest Regressor | Predict ROI value | `ROI` (continuous) | MAE: 1.50, R²: ~0.00 |
| Logistic Regression | Predict campaign success | `Success` (ROI > 5 = 1) | Accuracy: 49.7% |

### Features Used

```python
X = ['Clicks', 'Impressions', 'Engagement_Score', 'Duration_Days']
```

---

### Part A — Random Forest Regressor (ROI Prediction)

**Steps:**

1. **Load Data** — print column names to verify structure
2. **Define Features & Target** — `X` = 4 numeric features, `y` = `ROI`
3. **Train/Test Split** — 80% train / 20% test, `random_state=42`
4. **Train Model** — `RandomForestRegressor(n_estimators=50, max_depth=10)`
5. **Evaluate**
   - Mean Absolute Error: **1.50** — predictions are off by ~1.50 ROI units on average
   - R² Score: **~0.00** — the model does not explain variance in ROI well
6. **Save Model** — exported to `models/roi_model.pkl` using `joblib`
7. **Sample Prediction** — predict ROI for a custom campaign:
   ```python
   [Clicks=5000, Impressions=20000, Engagement_Score=75, Duration_Days=45]
   → Predicted ROI: 4.77
   ```
8. **Visualizations**
   - Actual vs Predicted scatter plot
   - Feature importance bar chart
   - Residual error scatter plot
   - Correlation heatmap (Clicks, Impressions, Engagement_Score, Duration_Days, ROI)
   - ROI distribution histogram

> **Note:** The low R² score suggests that `Clicks`, `Impressions`, `Engagement_Score`, and `Duration_Days` alone are **weak predictors of ROI**. Categorical features (channel, campaign type, audience) may improve model performance.

---

### Part B — Logistic Regression (Success Classification)

**Steps:**

1. **Create Binary Target** — label campaigns as successful if `ROI > 5`
   ```python
   df['Success'] = df['ROI'].apply(lambda x: 1 if x > 5 else 0)
   ```
2. **Define Features & Target** — same 4 numeric features, `y` = `Success`
3. **Train/Test Split** — 80/20 split
4. **Train Model** — `LogisticRegression()`
5. **Evaluate**
   - Accuracy: **49.7%** — barely better than random guessing (50%)
6. **Visualizations**
   - Confusion matrix heatmap (correct vs incorrect predictions)
   - ROC Curve with AUC score
   - Model accuracy bar chart

> **Note:** The near-random accuracy confirms that the selected numeric features have very low predictive power for campaign success. Future iterations should include engineered or categorical features.

---

### Feature Importance Insight

The Random Forest model reveals which inputs drive ROI predictions most:

| Feature | Importance |
|---|---|
| `Engagement_Score` | Highest |
| `Duration_Days` | Medium |
| `Impressions` | Medium |
| `Clicks` | Lowest |

---

## 🚀 How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Run Order

```
1. marketing_optimization.ipynb   ← Start here (EDA)
2. prediction_models.ipynb        ← Then run this (ML models)
```

Both notebooks read from: `../updated_marketing_campaign_dataset.csv`

---

## 🔑 Key Takeaways

- **Facebook** delivers the best ROI among all channels; **YouTube** is the cheapest to acquire customers
- **Display ads on Website targeting all genders** is the single best-performing campaign combination (ROI = 5.11)
- Campaign duration has minimal impact on ROI — shorter campaigns perform roughly the same as longer ones
- The numeric features (`Clicks`, `Impressions`, `Engagement_Score`, `Duration_Days`) are **weak predictors** of ROI on their own — categorical variables and feature engineering would significantly improve model accuracy

---

## 🛠️ Potential Improvements

- Add categorical features (channel, campaign type, gender) via one-hot encoding
- Try gradient boosting models (XGBoost, LightGBM)
- Perform hyperparameter tuning with GridSearchCV
- Add cross-validation for more robust evaluation
- Engineer interaction features (e.g. CTR = Clicks / Impressions)

---

*Dataset: `updated_marketing_campaign_dataset.csv`*
*Python 3.12 | scikit-learn | pandas | matplotlib | seaborn*
