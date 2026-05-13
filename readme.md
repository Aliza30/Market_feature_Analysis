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


# 🧠 Marketing Analytics — Analysis Overview

> A full-stack data science project combining **Exploratory Data Analysis**, **Segmentation Analysis**, **ROI Attribution**, **Predictive Modeling**, and **Classification** on real-world marketing campaign data.

---

## 📌 What This Project Is

This project is a **end-to-end marketing analytics pipeline** built to answer the question most businesses ask but rarely answer well:

> *"Where should we spend our marketing budget to get the highest return?"*

Rather than guessing based on intuition, this analysis uses **structured data science methodology** — from raw data cleaning all the way to trained machine learning models — to deliver evidence-based answers.

---

## 🔬 Types of Analysis Performed

### 1. 📊 Exploratory Data Analysis (EDA)
The foundation of any data science project. Before building models or drawing conclusions, the data was explored to understand its structure, distributions, and relationships.

- Inspected column types, value ranges, and missing patterns
- Previewed sample records to understand what each field represents
- Identified data quality issues (e.g. `Acquisition_Cost` stored as a formatted string `"$16,174.00"` instead of a numeric float)

---

### 2. 🧹 Data Cleaning & Feature Engineering
Raw data is never analysis-ready. Two key transformations were applied:

**Cleaning:**
- Stripped currency formatting (`$`, `,`) from `Acquisition_Cost` using regex
- Converted the column to `float` for numerical computation

**Feature Engineering:**
- Created a new categorical column `Duration_Category` by **binning** `Duration_Days` into meaningful groups (`Below 30`, `30-60`, `60+`)
- This transforms a continuous variable into an ordered categorical one — enabling group-level comparisons

**Why it matters:** Feature engineering directly impacts the quality of both analysis and model predictions. Without it, patterns stay hidden in raw numbers.

---

### 3. 📈 ROI Analysis
**Return on Investment (ROI) analysis** measures which marketing strategies generate the most value relative to cost.

Three dimensions of ROI were analyzed:

| Dimension | Method | Top Performer |
|---|---|---|
| Campaign Duration | GroupBy + Mean ROI | Below 30 days (ROI ≈ 5.00) |
| Marketing Channel | GroupBy + Mean ROI | Facebook (ROI ≈ 5.02) |
| Target Audience | GroupBy + Mean ROI | All genders (ROI ≈ 5.01) |

**Technique used:** `pandas groupby()` + `.mean()` aggregation, sorted and visualized as bar charts.

**Business value:** Tells the marketing team exactly which channels and audiences produce the highest return — allowing smarter budget allocation.

---

### 4. 🧩 Segmentation Analysis
Segmentation analysis divides a population into meaningful groups to understand behavioral or performance differences.

In this project, campaigns were segmented by:

- **Channel** (Facebook, YouTube, Google Ads, Instagram, Email, Website)
- **Campaign Type** (Display, Email, Influencer, Search)
- **Target Gender** (Men, Women, All)
- **Duration Band** (short, mid, long)

The key segmentation insight was the **multi-variable grouping** — combining all three dimensions (Campaign Type × Channel × Gender) to find the highest-performing segment combination:

```
Display + Website + All Genders → ROI = 5.11  ✅ Best overall segment
Influencer + Website + All     → ROI = 5.10
Display + Email + Women        → ROI = 5.09
```

**Technique used:** Multi-key `groupby()`, sorted aggregation, and a **pivot table heatmap** to visualize all segment combinations at once.

**Business value:** Moves beyond single-variable thinking. Instead of asking "what's the best channel?", it asks "what's the best *combination* of channel, campaign type, and audience?" — a far more actionable question.

---

### 5. 💰 Acquisition Cost Analysis
Cost efficiency is as important as ROI. A campaign can have a high ROI but also have a high cost to acquire each customer.

Average acquisition cost was calculated per channel and ranked:

| Rank | Channel | Avg. Cost |
|---|---|---|
| 1 (cheapest) | YouTube | $12,481 |
| 2 | Website | $12,488 |
| 3 | Instagram | $12,492 |
| 4 | Facebook | $12,511 |
| 5 | Email | $12,526 |
| 6 (most expensive) | Google Ads | $12,528 |

**Technique used:** GroupBy + Mean + sorted horizontal bar chart with a zoomed x-axis (`xlim`) to expose small but real cost differences.

**Business value:** Identifies where you can get the same result for less money — critical for cost-conscious campaign planning.

---

### 6. 🤖 Predictive Modeling — Regression (ROI Forecasting)
Shifts from *describing the past* to *predicting the future*.

**Model:** `RandomForestRegressor`
**Target:** Predict the continuous ROI value of a campaign before it runs

**Features used as inputs:**
```
Clicks, Impressions, Engagement_Score, Duration_Days
```

**Results:**

| Metric | Value | Interpretation |
|---|---|---|
| Mean Absolute Error | 1.50 | Predictions are off by ~1.5 ROI units on average |
| R² Score | ~0.00 | Model explains very little variance in ROI |

**What this tells us:** The numeric engagement features alone are **not strong predictors** of ROI. This is itself a data science insight — it means ROI is likely driven more by *categorical factors* (which channel, which campaign type, which audience) than by raw click and impression volume. This opens the door to model improvement.

**Model was saved** to `models/roi_model.pkl` using `joblib` for future use in production or APIs.

---

### 7. 🎯 Classification Modeling — Campaign Success Prediction
Instead of predicting the exact ROI value, the classification model asks a binary business question:

> *"Will this campaign succeed or fail?"*

**Target variable engineered:**
```python
Success = 1  if ROI > 5
Success = 0  if ROI ≤ 5
```

**Model:** `LogisticRegression`

**Evaluation:**

| Metric | Value | Interpretation |
|---|---|---|
| Accuracy | 49.7% | Barely above random chance (50%) |
| Confusion Matrix | Visualized | Shows false positive / false negative breakdown |
| ROC Curve + AUC | Plotted | Measures model's discrimination ability |

**What this tells us:** The same conclusion as the regression model — numeric features alone do not reliably separate successful from unsuccessful campaigns. The classification boundary (ROI > 5) is not captured by clicks, impressions, engagement, or duration alone.

---

### 8. 📉 Model Diagnostics
Beyond just reporting accuracy, model diagnostics were used to understand *how* and *where* the model fails:

- **Actual vs Predicted scatter plot** — reveals systematic over- or under-prediction
- **Residual Error plot** — checks for patterns in prediction errors (ideally, residuals should be random and centered at zero)
- **Feature Importance chart** — ranks which input features the Random Forest relied on most
- **Correlation Heatmap** — shows the linear relationships between all numeric features and ROI

---

## 💡 What This Analysis Brings to the Table

### For the Business
| Insight | Action |
|---|---|
| Facebook has the highest ROI | Prioritize Facebook budget allocation |
| YouTube has the lowest acquisition cost | Use YouTube for cost-efficient reach |
| Display + Website + All Genders is the top combo | Default campaign template for new launches |
| Short campaigns (< 30 days) perform as well as long ones | Avoid over-investing in long campaign durations |
| Numeric engagement signals don't predict ROI well | Don't optimize solely for clicks and impressions |

### For the Data Science Team
| Finding | Next Step |
|---|---|
| R² ≈ 0, Accuracy ≈ 50% | Add categorical features via one-hot encoding |
| Feature importance: Engagement > Duration > Impressions > Clicks | Drop clicks or use CTR (Clicks/Impressions) instead |
| ROI distribution is roughly uniform | Try stratified sampling to balance the classification target |
| Model saved as `.pkl` | Ready to wrap in a Flask/FastAPI endpoint for real-time predictions |

---

## 🗂️ Analysis Type Summary

| Analysis | Category | Technique |
|---|---|---|
| Data inspection & cleaning | EDA | `df.head()`, regex, `astype()` |
| Duration bucketing | Feature Engineering | `pd.cut()` |
| ROI by channel / duration / gender | ROI Analysis | `groupby().mean()` |
| Best campaign combination | Segmentation Analysis | Multi-key `groupby()` + pivot heatmap |
| Acquisition cost ranking | Cost Efficiency Analysis | `groupby().mean()` + sorted bar chart |
| ROI forecasting | Regression (Supervised ML) | `RandomForestRegressor` |
| Success/fail prediction | Classification (Supervised ML) | `LogisticRegression` |
| Error diagnostics | Model Evaluation | MAE, R², Confusion Matrix, ROC/AUC |
| Input-output relationships | Correlation Analysis | `seaborn` heatmap |
| Prediction uncertainty | Residual Analysis | Residual scatter plot |

---

## 🧰 Tools & Libraries

| Tool | Role |
|---|---|
| `pandas` | Data loading, cleaning, groupby, pivot |
| `numpy` | Numerical operations |
| `matplotlib` | Charts and plots |
| `seaborn` | Heatmaps and styled visualizations |
| `scikit-learn` | ML models, train/test split, metrics |
| `joblib` | Model serialization (.pkl) |

---

## 🏁 Summary

This project demonstrates a complete data science workflow applied to a marketing use case:

```
Raw Data → Cleaning → EDA → Segmentation → ROI Analysis
       → Feature Engineering → ML Modeling → Evaluation → Insights
```

The analysis does not just describe what happened — it builds tools to **predict what will happen**, and delivers **clear, prioritized business recommendations** backed by data.
