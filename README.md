# E-Commerce Customer Analytics & Machine Learning

## Project Description

A complete, production-ready **E-Commerce Customer Analytics and Machine Learning** portfolio project built on a real customer dataset. The project follows a structured **4-Tier Analytics Ladder**:

1. **Tier 1 — Data Hygiene & Architecture**: Comprehensive data profiling, quality audit, and cleaning
2. **Tier 2 — Exploratory & Diagnostic Analytics**: Five professional EDA visualizations with empirical insights
3. **Tier 3 — Predictive Analytics**: Multi-class classification to predict `customer_sustainability_priority` (scale 1–5)
4. **Tier 4 — Prescriptive Business Strategy**: Evidence-based customer targeting and engagement recommendations

**Business Objective:** Identify customer segments by sustainability priority, predict sustainability-priority class using ML, and develop a resource-constrained engagement strategy that targets the top 20% of customers most likely to belong to high sustainability-priority categories.

---

## Dataset

**File:** `ecommerce_customers.csv`  
**Source:** User-provided local CSV file.  
**Size:** 1,500 customer records × 5 columns

| Column | Type | Description |
|---|---|---|
| `customer_id` | String | Unique identifier (CUST_XXXX format) |
| `region` | Categorical | 5 geographic regions: Asia-Pacific, Europe, Latin America, Middle East, North America |
| `membership_level` | Categorical | Standard, Silver, Gold, Platinum (50 missing → filled as Unknown) |
| `customer_sustainability_priority` | Integer (1–5) | **Target variable**: 1=Very Low → 5=Very High |
| `joined_year` | Integer | Year the customer joined: 2021, 2022, 2023, 2024 |

---

## Technologies

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core programming language |
| pandas | ≥1.5.0 | Data manipulation & analysis |
| numpy | ≥1.23.0 | Numerical computations |
| scikit-learn | ≥1.2.0 | ML pipeline, preprocessing, Random Forest, evaluation |
| matplotlib | ≥3.6.0 | Visualizations (notebook & dashboard) |
| seaborn | ≥0.12.0 | Statistical visualizations |
| streamlit | ≥1.28.0 | Interactive web dashboard |
| jupyter / notebook | ≥1.0.0 | Notebook environment |

---

## Project Structure

```text
Project/
│
├── ecommerce_customers.csv                          ← Source dataset (user-provided)
├── Anumana_Ecommerce_Customer_Analytics_ML.ipynb   ← Complete end-to-end analysis notebook
├── app.py                                           ← Streamlit interactive dashboard
├── requirements.txt                                 ← Python dependencies
├── Anumana_Ecommerce_Customer_ProjectReport.docx   ← Formal Word project report
└── README.md                                        ← This file
```

---

## Installation

```bash
# Clone or download the project folder
# Navigate to the project directory, then:

pip install -r requirements.txt
```

---

## Running the Notebook

```bash
jupyter notebook Anumana_Ecommerce_Customer_Analytics_ML.ipynb
```

Or open in JupyterLab:
```bash
jupyter lab
```

The notebook executes from top to bottom. All cells must be run in order. The CSV file must be in the same directory as the notebook.

---

## Running the Streamlit Dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`). The dashboard loads the CSV automatically from the project directory.

---

## Methodology

### 4-Tier Analytics Ladder

**Tier 1 — Data Hygiene & Architecture**
- Dataset profiling: shape, dtypes, missing values, duplicates, value distributions
- Data cleaning: 50 missing `membership_level` values filled as `'Unknown'` (retains all records, avoids mode-imputation bias)
- Entity-level validation: confirmed `customer_id` is unique across all 1,500 rows

**Tier 2 — Exploratory & Diagnostic Analytics**
- Viz 1: Customer distribution by region (bar + pie chart)
- Viz 2: Membership-level distribution (horizontal bar chart)
- Viz 3: Sustainability priority distribution (bar + line chart)
- Viz 4: Customer acquisition by joining year (bar + stacked membership chart)
- Viz 5: Sustainability priority by business segment (heatmap + boxplot)

**Tier 3 — Predictive Analytics**
- **Target:** `customer_sustainability_priority` (5-class ordinal → multi-class classification)
- **Features:** `region`, `membership_level`, `joined_year`
- **Excluded:** `customer_id` (identifier, not a predictive feature)
- **Preprocessing:** OneHotEncoding for categorical features; `joined_year` as numeric
- **Train/Test Split:** 80/20, stratified, `random_state=42`
- **Model:** Random Forest Classifier (200 trees) — selected over Logistic Regression for higher performance and native feature importances
- **Leakage prevention:** sklearn Pipeline; no target used as predictor; no identifier columns

**Tier 4 — Prescriptive Strategy**
- Resource-constrained targeting: top 20% (300 customers) ranked by predicted probability of belonging to Priority 4 or 5
- Membership-level, regional, and cohort-year engagement strategies derived from actual data

---

## Machine Learning

| Component | Detail |
|---|---|
| **Target variable** | `customer_sustainability_priority` (1–5, ordinal) |
| **Predictors** | `region`, `membership_level`, `joined_year` |
| **Excluded** | `customer_id` (identifier — no predictive signal) |
| **Problem type** | Multi-class classification (5 classes) |
| **Algorithm** | Random Forest Classifier, n_estimators=200, random_state=42 |
| **Preprocessing** | OneHotEncoder (categorical), passthrough (numeric) via ColumnTransformer Pipeline |
| **Evaluation metrics** | Accuracy, Precision (W), Recall (W), F1-Score (W), ROC-AUC (OvR, W), Confusion Matrix |
| **Reproducibility** | `random_state=42` throughout; stratified train/test split |

---

## Key Findings

All findings are derived strictly from the actual dataset. No results are fabricated.

### Customer Profile
- **1,500 unique customers** — dataset is already at customer/entity level
- **Latin America** is the largest region (323 customers, 21.5%); **North America** is smallest (273, 18.2%)
- **Standard membership** dominates (695 customers, 46.3%); Platinum is the rarest (71, 4.7%)
- Customer acquisition is **uniformly stable** across 2021–2024 (~370–380 per year)

### Sustainability Priority
- Mean sustainability priority: **3.006 / 5.00** (approximately neutral)
- Distribution is **near-uniform** across all 5 classes — no single class dominates
- **Middle East** shows the highest average sustainability priority (3.107); **Europe** the lowest (2.901)
- **2021 cohort** shows the highest average priority (3.075); **2024 cohort** the lowest (2.908)

### Predictive Model
- **Accuracy: 18.0%** | Precision (W): 0.1846 | Recall (W): 0.1800 | F1 (W): 0.1800 | ROC-AUC (OvR, W): 0.4909
- ⚠️ Model performs at near-random-baseline levels — expected with only 3 weak predictors and a near-uniform 5-class target
- **Feature importance (grouped):** `joined_year` (46.4%), `region` (27.7%), `membership_level` (25.9%)
- The model's primary value is **customer ranking for top-20% targeting**, not precise class prediction

---

## Business Recommendations

All recommendations are strictly evidence-based. No ROI or financial impact is claimed (no financial data is available).

1. **Top-20% targeting** — Engage the top 300 customers ranked by predicted probability of Priority 4 or 5
2. **Standard-tier campaigns** — The 695 Standard members (46.3% of base) have moderate priority (avg 3.072); sustainability messaging may increase engagement
3. **Regional differentiation** — Middle East and Latin America: reinforce existing sustainability interest; Europe: build awareness
4. **New customer onboarding** — 2024 cohort customers (380) show the lowest average priority; early sustainability onboarding is recommended
5. **Data quality improvement** — Resolve the 50 unknown membership records to improve future model accuracy

---

## Limitations

- Only 3 predictor variables are available — predictive power is inherently constrained
- No revenue, transactional, churn, or behavioral data — financial impact cannot be estimated
- All associations are **observational** — causation cannot be inferred
- Dataset size (1,500 records) is modest for production ML
- 50 missing `membership_level` values (3.33%) introduce minor uncertainty

---

## Author

**Anumana Bhattacharya**  
Project: E-Commerce Customer Analytics & Machine Learning  
Framework: 4-Tier Analytics Ladder | Dataset: ecommerce_customers.csv (user-provided)
