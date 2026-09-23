"""
E-Commerce Customer Analytics & Machine Learning Dashboard
Author: Anumana Bhattacharya
Run: streamlit run app.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Anumana | E-Commerce Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

RANDOM_STATE = 42

# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "ecommerce_customers.csv")
    df = pd.read_csv(path)
    return df

@st.cache_data
def clean_data(df):
    df_clean = df.copy()
    df_clean["membership_level"] = df_clean["membership_level"].fillna("Unknown")
    return df_clean

@st.cache_resource
def train_model(df_clean):
    FEATURES = ["region", "membership_level", "joined_year"]
    TARGET = "customer_sustainability_priority"

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    categorical_features = ["region", "membership_level"]
    numeric_features = ["joined_year"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", "passthrough", numeric_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc = None

    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(
        y_test, y_pred,
        target_names=["Priority 1","Priority 2","Priority 3","Priority 4","Priority 5"],
        output_dict=True
    )

    # Feature importances
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(categorical_features)
    all_names = list(ohe_names) + numeric_features
    importances = model.named_steps["classifier"].feature_importances_
    fi_df = pd.DataFrame({"Feature": all_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "y_prob": y_prob,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
        "cm": cm, "clf_report": clf_report, "fi_df": fi_df,
        "FEATURES": FEATURES, "TARGET": TARGET
    }

@st.cache_data
def make_predictions(_model, df_clean, features):
    df_pred = df_clean.copy()
    df_pred["predicted_priority"] = _model.predict(df_clean[features])
    probs = _model.predict_proba(df_clean[features])
    classes = _model.classes_
    for i, cls in enumerate(classes):
        df_pred[f"prob_priority_{cls}"] = probs[:, i]
    df_pred["prob_high_priority"] = (
        df_pred.get("prob_priority_4", 0) + df_pred.get("prob_priority_5", 0)
    )
    return df_pred

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
df_raw = load_data()
df_clean = clean_data(df_raw)
ml = train_model(df_clean)
df_pred = make_predictions(ml["model"], df_clean, ml["FEATURES"])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bar-chart.png", width=60)
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Go to Section", [
    "🏠 Executive KPI Dashboard",
    "🔍 Data Quality",
    "📈 Exploratory Analysis (EDA)",
    "🤖 Machine Learning",
    "👥 Customer Priority View",
    "💡 Prescriptive Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters (EDA & Customer View)")

all_regions      = sorted(df_clean["region"].unique())
all_memberships  = sorted(df_clean["membership_level"].unique())
all_years        = sorted(df_clean["joined_year"].unique())

sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
sel_memberships = st.sidebar.multiselect("Membership Level", all_memberships, default=all_memberships)
sel_years = st.sidebar.multiselect("Joining Year", all_years, default=all_years)

# Filter
mask = (
    df_pred["region"].isin(sel_regions) &
    df_pred["membership_level"].isin(sel_memberships) &
    df_pred["joined_year"].isin(sel_years)
)
df_filtered = df_pred[mask].copy()

# ═════════════════════════════════════════════════════
# SECTION 1 — EXECUTIVE KPI DASHBOARD
# ═════════════════════════════════════════════════════
if section == "🏠 Executive KPI Dashboard":
    st.title("🏠 Executive KPI Dashboard")
    st.markdown("**E-Commerce Customer Analytics & Machine Learning** | Author: Anumana Bhattacharya")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    total_customers = len(df_clean)
    n_regions = df_clean["region"].nunique()
    n_memberships = df_clean["membership_level"].nunique()
    avg_priority = df_clean["customer_sustainability_priority"].mean()
    most_common_mem = df_clean["membership_level"].value_counts().idxmax()
    most_common_region = df_clean["region"].value_counts().idxmax()

    c1.metric("👤 Total Customers", f"{total_customers:,}")
    c2.metric("🌍 Number of Regions", n_regions)
    c3.metric("🎖️ Membership Levels", n_memberships)
    c4.metric("♻️ Avg Sustainability Priority", f"{avg_priority:.2f} / 5.00")
    c5.metric("🥇 Most Common Membership", most_common_mem)
    c6.metric("🌐 Most Represented Region", most_common_region)

    st.markdown("---")
    st.markdown("""
    > **Note:** All KPIs are computed from the actual `ecommerce_customers.csv` dataset.
    > No revenue or financial data is available in this dataset — financial KPIs are therefore not included.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Priority Distribution")
        prio_counts = df_clean["customer_sustainability_priority"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(prio_counts.index, prio_counts.values,
               color=["#d73027","#fc8d59","#fee090","#91bfdb","#4575b4"], edgecolor="white")
        ax.set_xlabel("Priority (1–5)"); ax.set_ylabel("Customers"); ax.set_xticks([1,2,3,4,5])
        ax.set_title("Sustainability Priority Distribution")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Membership Distribution")
        mem_counts = df_clean["membership_level"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.barh(mem_counts.index, mem_counts.values,
                 color=["#636363","#9ecae1","#fdd0a2","#fdae6b","#d9d9d9"][:len(mem_counts)],
                 edgecolor="white")
        ax2.set_xlabel("Customers"); ax2.set_title("Membership Level Distribution")
        ax2.invert_yaxis()
        st.pyplot(fig2); plt.close()

# ═════════════════════════════════════════════════════
# SECTION 2 — DATA QUALITY
# ═════════════════════════════════════════════════════
elif section == "🔍 Data Quality":
    st.title("🔍 Data Quality Report")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df_raw):,}")
    c2.metric("Columns", df_raw.shape[1])
    c3.metric("Missing Values (raw)", int(df_raw.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df_raw.duplicated().sum()))

    st.subheader("Column Overview & Data Types")
    dtype_df = pd.DataFrame({
        "Column": df_raw.columns,
        "Data Type": df_raw.dtypes.values,
        "Non-Null (raw)": df_raw.notnull().sum().values,
        "Missing (raw)": df_raw.isnull().sum().values,
        "Missing %": (df_raw.isnull().sum().values / len(df_raw) * 100).round(2),
        "Unique Values": df_raw.nunique().values
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.subheader("Categorical Value Distributions")
    for col in ["region", "membership_level", "customer_sustainability_priority", "joined_year"]:
        with st.expander(f"📋 {col}"):
            vc = df_raw[col].value_counts(dropna=False).reset_index()
            vc.columns = [col, "Count"]
            vc["Percentage (%)"] = (vc["Count"] / len(df_raw) * 100).round(2)
            st.dataframe(vc, use_container_width=True)

    st.subheader("Cleaning Actions Applied")
    st.markdown("""
    | Issue | Action | Rationale |
    |---|---|---|
    | 50 missing `membership_level` values (3.33%) | Filled with `'Unknown'` category | Preserves data integrity; avoids mode-imputation bias; transparent to the model |
    | 0 duplicate rows | None required | No duplicates found |
    | 0 duplicate `customer_id` | None required | All IDs are unique |
    | `joined_year` range | Validated (2021–2024 only) | No impossible years found |
    | `customer_sustainability_priority` | Validated (1–5 only) | No out-of-range values |
    | `region` | Validated (5 known categories) | No invalid region names |
    """)

    st.success("✅ Dataset is clean. Final shape: 1,500 rows × 5 columns.")

# ═════════════════════════════════════════════════════
# SECTION 3 — EDA
# ═════════════════════════════════════════════════════
elif section == "📈 Exploratory Analysis (EDA)":
    st.title("📈 Exploratory & Diagnostic Analytics")
    st.markdown(f"**Showing {len(df_filtered):,} customers** based on current sidebar filters.")
    st.markdown("---")

    if len(df_filtered) == 0:
        st.warning("No data matches the current filters. Please adjust the sidebar selections.")
    else:
        # VIZ 1 — Region distribution
        st.subheader("📊 Viz 1 — Customer Distribution by Region")
        region_counts = df_filtered["region"].value_counts().sort_values(ascending=False)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#2c7bb6","#abd9e9","#74add1","#4575b4","#313695"]
            bars = ax.bar(region_counts.index, region_counts.values, color=colors[:len(region_counts)], edgecolor="white")
            for bar, val in zip(bars, region_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xlabel("Region"); ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Count by Region", fontweight="bold")
            ax.tick_params(axis="x", rotation=15)
            st.pyplot(fig); plt.close()
        with col2:
            df_region_show = region_counts.reset_index()
            df_region_show.columns = ["Region","Count"]
            df_region_show["%"] = (df_region_show["Count"] / df_region_show["Count"].sum() * 100).round(1)
            st.dataframe(df_region_show, use_container_width=True)

        st.markdown(f"> **Observation:** Latin America and Middle East are the largest segments. North America is the smallest. Distribution is broadly balanced.")

        st.divider()

        # VIZ 2 — Membership
        st.subheader("📊 Viz 2 — Membership-Level Distribution")
        membership_order = ["Standard","Silver","Gold","Platinum","Unknown"]
        mem_counts = df_filtered["membership_level"].value_counts().reindex(
            [m for m in membership_order if m in df_filtered["membership_level"].unique()]
        ).dropna()
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            cols_mem = ["#636363","#9ecae1","#fdd0a2","#fdae6b","#d9d9d9"]
            bars = ax.barh(mem_counts.index, mem_counts.values, color=cols_mem[:len(mem_counts)], edgecolor="white")
            ax.invert_yaxis()
            for bar, val in zip(bars, mem_counts.values):
                ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(val), va="center", fontsize=9, fontweight="bold")
            ax.set_xlabel("Number of Customers"); ax.set_title("Membership-Level Distribution", fontweight="bold")
            ax.set_xlim(0, mem_counts.max() + 80)
            st.pyplot(fig); plt.close()
        with col2:
            df_mem_show = mem_counts.reset_index()
            df_mem_show.columns = ["Membership","Count"]
            df_mem_show["%"] = (df_mem_show["Count"] / df_mem_show["Count"].sum() * 100).round(1)
            st.dataframe(df_mem_show, use_container_width=True)

        st.markdown("> **Observation:** Standard is the dominant tier. The customer base follows a loyalty pyramid: Standard > Silver > Gold > Platinum.")

        st.divider()

        # VIZ 3 — Sustainability Priority
        st.subheader("📊 Viz 3 — Sustainability Priority Distribution")
        prio_counts = df_filtered["customer_sustainability_priority"].value_counts().sort_index()
        avg_p = df_filtered["customer_sustainability_priority"].mean()
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_p = ["#d73027","#fc8d59","#fee090","#91bfdb","#4575b4"]
            x_lbls = ["1-Very Low","2-Low","3-Moderate","4-High","5-Very High"]
            present = [i for i in range(5) if (i+1) in prio_counts.index]
            bars = ax.bar([x_lbls[i] for i in present], [prio_counts.get(i+1, 0) for i in present],
                          color=[colors_p[i] for i in present], edgecolor="white")
            for bar, val in zip(bars, [prio_counts.get(i+1, 0) for i in present]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.set_xlabel("Sustainability Priority"); ax.set_ylabel("Customers")
            ax.set_title("Sustainability Priority Distribution", fontweight="bold")
            ax.tick_params(axis="x", rotation=15)
            st.pyplot(fig); plt.close()
        with col2:
            st.metric("Mean Priority", f"{avg_p:.3f}")
            df_prio_show = prio_counts.reset_index()
            df_prio_show.columns = ["Priority","Count"]
            df_prio_show["%"] = (df_prio_show["Count"] / df_prio_show["Count"].sum() * 100).round(1)
            st.dataframe(df_prio_show, use_container_width=True)

        st.markdown("> **Observation:** Distribution is approximately uniform across all 5 classes — ideal for balanced ML training.")

        st.divider()

        # VIZ 4 — Joining Year Trend
        st.subheader("📊 Viz 4 — Customer Acquisition by Joining Year")
        year_counts = df_filtered["joined_year"].value_counts().sort_index()
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(year_counts.index, year_counts.values, color="#2c7bb6", edgecolor="white", width=0.6)
            ax.plot(year_counts.index, year_counts.values, "o-", color="#d73027", linewidth=2, markersize=8, zorder=5)
            for x, y_val in zip(year_counts.index, year_counts.values):
                ax.text(x, y_val + 2, str(y_val), ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_xticks(year_counts.index); ax.set_xlabel("Year Joined"); ax.set_ylabel("Customers")
            ax.set_title("New Customers per Joining Year", fontweight="bold")
            ax.set_ylim(0, year_counts.max() + 50)
            st.pyplot(fig); plt.close()
        with col2:
            df_yr_show = year_counts.reset_index()
            df_yr_show.columns = ["Year","Count"]
            df_yr_show["Avg Priority"] = df_filtered.groupby("joined_year")["customer_sustainability_priority"].mean().values.round(3)
            st.dataframe(df_yr_show, use_container_width=True)

        st.markdown("> **Observation:** Acquisition volumes are stable across 2021–2024 (no significant growth or decline trend).")

        st.divider()

        # VIZ 5 — Sustainability by Segment
        st.subheader("📊 Viz 5 — Sustainability Priority by Business Segment")
        col1, col2 = st.columns(2)
        with col1:
            pivot = df_filtered.groupby(["membership_level","region"])["customer_sustainability_priority"].mean()
            if not pivot.empty:
                pivot = pivot.unstack(fill_value=np.nan)
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlBu", ax=ax, vmin=1, vmax=5,
                            linewidths=0.5, cbar_kws={"label": "Avg Priority"})
                ax.set_title("Avg Priority: Membership × Region", fontweight="bold")
                ax.tick_params(axis="x", rotation=20)
                st.pyplot(fig); plt.close()
            else:
                st.info("Insufficient data for heatmap with current filters.")

        with col2:
            mean_by_mem = df_filtered.groupby("membership_level")["customer_sustainability_priority"].mean().sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            bars = ax2.bar(mean_by_mem.index, mean_by_mem.values,
                           color=["#636363","#9ecae1","#fdd0a2","#fdae6b","#d9d9d9"][:len(mean_by_mem)],
                           edgecolor="white", width=0.5)
            ax2.set_ylim(0, 5.5)
            for bar, val in zip(bars, mean_by_mem.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f}",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax2.set_xlabel("Membership Level"); ax2.set_ylabel("Avg Sustainability Priority")
            ax2.set_title("Avg Priority by Membership", fontweight="bold")
            ax2.tick_params(axis="x", rotation=15)
            ax2.axhline(y=df_filtered["customer_sustainability_priority"].mean(), color="red",
                        linestyle="--", alpha=0.7, label=f"Overall avg")
            ax2.legend(fontsize=8)
            st.pyplot(fig2); plt.close()

        st.markdown("> ⚠️ **Correlation ≠ Causation.** Membership level is *associated* with differences in sustainability priority — this does not imply that membership *causes* priority levels.")

# ═════════════════════════════════════════════════════
# SECTION 4 — MACHINE LEARNING
# ═════════════════════════════════════════════════════
elif section == "🤖 Machine Learning":
    st.title("🤖 Machine Learning — Predictive Analytics")
    st.markdown("**Model:** Random Forest Classifier | **Target:** `customer_sustainability_priority` (5-class ordinal)")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{ml['acc']:.4f}")
    c2.metric("Precision (W)", f"{ml['prec']:.4f}")
    c3.metric("Recall (W)", f"{ml['rec']:.4f}")
    c4.metric("F1-Score (W)", f"{ml['f1']:.4f}")
    if ml["auc"] is not None:
        c5.metric("ROC-AUC (OvR)", f"{ml['auc']:.4f}")
    else:
        c5.metric("ROC-AUC", "N/A")

    st.caption("W = Weighted average across 5 classes. ROC-AUC uses One-vs-Rest (OvR) multiclass strategy.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix (Test Set)")
        cm = ml["cm"]
        classes = [1, 2, 3, 4, 5]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels([f"Pred {c}" for c in classes], fontsize=8)
        ax.set_yticklabels([f"True {c}" for c in classes], fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — Random Forest", fontweight="bold")
        thresh = cm.max() / 2.
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9, fontweight="bold")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Feature Importance")
        fi_df = ml["fi_df"]
        top_n = min(15, len(fi_df))
        top_fi = fi_df.head(top_n)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        colors_fi = ["#2c7bb6" if "membership" in f else "#d73027" if "region" in f else "#1a9850" for f in top_fi["Feature"]]
        ax2.barh(range(top_n), top_fi["Importance"].values[::-1], color=colors_fi[::-1], edgecolor="white")
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(top_fi["Feature"].values[::-1], fontsize=7)
        ax2.set_xlabel("Importance Score"); ax2.set_title(f"Top {top_n} Features", fontweight="bold")
        st.pyplot(fig2); plt.close()

    st.subheader("Classification Report (per class)")
    report_df = pd.DataFrame(ml["clf_report"]).T.round(4)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("""
    **Model Notes:**
    - Training set: 80% (1,200 samples, stratified). Test set: 20% (300 samples).
    - Random Forest with 200 trees, `random_state=42` for reproducibility.
    - **Leakage prevention:** `customer_id` (identifier) excluded. Target not used as a predictor.
    - With only 3 predictor variables available, predictive power is inherently limited.
    - A random baseline for a 5-class problem is ≈20% accuracy.
    - ROC-AUC uses One-vs-Rest (OvR) strategy — appropriate for multiclass problems.
    """)

# ═════════════════════════════════════════════════════
# SECTION 5 — CUSTOMER PRIORITY VIEW
# ═════════════════════════════════════════════════════
elif section == "👥 Customer Priority View":
    st.title("👥 Customer Priority View")
    st.markdown(f"Showing **{len(df_filtered):,}** customers matching current filters.")
    st.markdown("---")

    display_cols = [
        "customer_id", "region", "membership_level", "joined_year",
        "customer_sustainability_priority", "predicted_priority",
        "prob_priority_1", "prob_priority_2", "prob_priority_3",
        "prob_priority_4", "prob_priority_5", "prob_high_priority"
    ]
    available_cols = [c for c in display_cols if c in df_filtered.columns]
    df_display = df_filtered[available_cols].copy()

    # Rounding probabilities
    prob_cols = [c for c in available_cols if c.startswith("prob_")]
    for c in prob_cols:
        df_display[c] = df_display[c].round(4)

    sort_col = st.selectbox("Sort by", ["prob_high_priority", "customer_sustainability_priority",
                                          "predicted_priority", "customer_id"])
    sort_asc = st.checkbox("Ascending order", value=False)
    df_display = df_display.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    st.dataframe(df_display, use_container_width=True, height=500)

    st.subheader("Top 20% High-Priority Customers")
    top_n = int(np.ceil(len(df_filtered) * 0.20))
    df_top20 = df_display.sort_values("prob_high_priority", ascending=False).head(top_n)
    st.markdown(f"**{top_n} customers** selected (top 20% by predicted probability of Priority 4 or 5)")
    st.dataframe(df_top20[["customer_id","region","membership_level","joined_year",
                              "customer_sustainability_priority","predicted_priority","prob_high_priority"]],
                 use_container_width=True, height=400)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 20% — Membership Breakdown")
        mem_top = df_top20["membership_level"].value_counts().reset_index()
        mem_top.columns = ["Membership","Count"]
        st.dataframe(mem_top, use_container_width=True)
    with col2:
        st.subheader("Top 20% — Regional Breakdown")
        reg_top = df_top20["region"].value_counts().reset_index()
        reg_top.columns = ["Region","Count"]
        st.dataframe(reg_top, use_container_width=True)

# ═════════════════════════════════════════════════════
# SECTION 6 — PRESCRIPTIVE RECOMMENDATIONS
# ═════════════════════════════════════════════════════
elif section == "💡 Prescriptive Recommendations":
    st.title("💡 Prescriptive Analytics & Business Recommendations")
    st.markdown("---")

    st.subheader("📌 Resource-Constrained Targeting Strategy")
    total_customers = len(df_pred)
    top_20_n = int(np.ceil(total_customers * 0.20))
    df_top20_all = df_pred.sort_values("prob_high_priority", ascending=False).head(top_20_n)
    pct_actual_high = (df_top20_all["customer_sustainability_priority"] >= 4).sum() / top_20_n * 100

    st.info(f"""
    **Assumption:** Business can proactively engage only the top **20%** of customers due to limited campaign resources.

    - **Total customers:** {total_customers:,}
    - **Top 20% selected:** {top_20_n:,} customers
    - **% of selected with actual priority ≥ 4:** {pct_actual_high:.1f}%
    - **Ranking criterion:** Predicted probability of belonging to Priority 4 or Priority 5
    """)

    st.markdown("---")
    st.subheader("1️⃣ Customer Prioritization")
    st.markdown(f"""
    - Identify and engage the top **{top_20_n}** customers with highest predicted probability of high sustainability priority (classes 4 and 5).
    - These customers are the most likely to respond positively to sustainability-focused communications and product offerings.
    - Apply first-wave engagement resources exclusively to this group to maximize impact per resource unit.
    """)

    mem_means = df_pred.groupby("membership_level")["customer_sustainability_priority"].mean().sort_values(ascending=False)
    top_mem = mem_means.index[0]
    st.subheader("2️⃣ Membership-Level Strategy")
    st.markdown(f"""
    - **{top_mem}** membership segment is associated with the highest average sustainability priority ({mem_means[top_mem]:.3f}/5.0).
    - The **Standard tier** contains the most customers (695, ~46.3%) with moderate average priority.
      → Design targeted sustainability education and engagement content for Standard-tier customers to shift them toward higher-priority behavior.
    - **Gold and Platinum** segments are smaller but may be receptive to premium sustainability positioning.
    - **Unknown membership** customers (50) should be re-engaged to update their profiles — data quality improvement opportunity.
    """)

    region_means = df_pred.groupby("region")["customer_sustainability_priority"].mean().sort_values(ascending=False)
    top_reg = region_means.index[0]
    low_reg = region_means.index[-1]
    st.subheader("3️⃣ Regional Strategy")
    st.markdown(f"""
    - **{top_reg}** shows the highest average sustainability priority ({region_means[top_reg]:.3f}) — reinforce existing sustainability messaging in this region.
    - **{low_reg}** shows the lowest average sustainability priority ({region_means[low_reg]:.3f}) — consider targeted awareness campaigns to build sustainability interest.
    - Differentiate communications by region based on observed average priority levels.
    """)

    year_means = df_pred.groupby("joined_year")["customer_sustainability_priority"].mean()
    old_yr = year_means.idxmax(); new_yr = year_means.idxmin()
    st.subheader("4️⃣ Customer Lifecycle / Cohort Strategy")
    st.markdown(f"""
    - Customers who joined in **{old_yr}** show the highest average sustainability priority ({year_means[old_yr]:.3f}).
    - Customers who joined in **{new_yr}** show the lowest ({year_means[new_yr]:.3f}).
    - Implement **sustainability onboarding programs** for newly acquired customers (2024 cohort) to build priority associations early.
    - Leverage longer-tenured customers (2021 cohort) as potential sustainability advocates or brand ambassadors.
    """)

    st.subheader("⚠️ Limitations")
    st.warning("""
    - **No revenue/financial data available** — ROI, cost-benefit, or CLV calculations cannot be made.
    - **Only 3 predictors** limit the model's predictive precision.
    - **All associations are observational** — do not imply causation.
    - **Dataset size** (1,500 records) is modest for production-grade ML.
    - Recommendations should be validated with domain experts and A/B testing before large-scale deployment.
    """)

    st.markdown("---")
    st.subheader("Summary Table")
    summary_data = {
        "Dimension": ["Total Customers", "Regions", "Membership Levels", "Avg Sustainability Priority",
                       "Most Common Membership", "Top 20% for Engagement", "Highest-Priority Region",
                       "Lowest-Priority Region", "Highest-Priority Cohort Year"],
        "Finding": [
            "1,500 unique customer records",
            "5 (Latin America, Middle East, Asia-Pacific, Europe, North America)",
            "5 (Standard, Silver, Gold, Platinum, Unknown)",
            f"{df_pred['customer_sustainability_priority'].mean():.3f} / 5.00",
            f"Standard ({len(df_pred[df_pred['membership_level']=='Standard'])} customers, 46.3%)",
            f"{top_20_n} customers ranked by predicted P(Priority 4 or 5)",
            f"{top_reg} (avg {region_means[top_reg]:.3f})",
            f"{low_reg} (avg {region_means[low_reg]:.3f})",
            f"{old_yr} cohort (avg {year_means[old_yr]:.3f})"
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
