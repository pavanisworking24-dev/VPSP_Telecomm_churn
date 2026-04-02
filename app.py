import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from scipy.stats import loguniform, randint, uniform
import shap

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📡 Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #1f3c88;
        border-bottom: 3px solid #DD8452; padding-bottom: 8px; margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #1f3c88;
        background: linear-gradient(90deg, #e8f0fe, transparent);
        padding: 6px 12px; border-left: 4px solid #DD8452;
        border-radius: 4px; margin: 16px 0 10px 0;
    }
    .metric-card {
        background: #f8f9ff; border: 1px solid #dde3f0;
        border-radius: 10px; padding: 12px 16px; text-align: center;
    }
    .metric-val { font-size: 1.6rem; font-weight: 800; color: #1f3c88; }
    .metric-lbl { font-size: 0.75rem; color: #666; margin-top: 2px; }
    .risk-high { background:#ffe0e0; border-left:4px solid #e74c3c;
                 padding:14px; border-radius:6px; margin-top:10px; }
    .risk-low  { background:#e0ffe0; border-left:4px solid #27ae60;
                 padding:14px; border-radius:6px; margin-top:10px; }
    div[data-testid="stSidebar"] { background: #1a2744; }
    div[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn ML App")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Home & Data Upload",
        "📊 EDA",
        "⚙️ Feature Engineering",
        "🤖 Train Models",
        "📈 Model Comparison",
        "🎯 Threshold Tuning",
        "🔍 SHAP Explainability",
        "🔮 Predict Single Customer"
    ])
    st.markdown("---")
    st.markdown("**Pipeline Steps**")
    for s in ["Data Load","EDA","Feature Eng.","Encode & Scale",
              "Feature Select","SMOTE","CV + Tuning","Evaluate","SHAP"]:
        st.markdown(f"✅ {s}")

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "df_raw": None, "df_feat": None, "models_trained": False,
    "results_df": None, "X_test_sel": None, "y_test": None,
    "tuned_models": None, "selected_features": None,
    "y_prob_dict": None, "scaler": None, "l1_mask": None,
    "feature_names": None, "X_train_res": None, "y_train_res": None,
    "thresh_df": None, "best_model_name": None, "cv_results": None,
    "shap_vals": None, "explainer": None,
    "X_test_sel_df": None, "X_train_for_pred": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# ─────────────────────────────────────────────
#  SHARED: FEATURE ENGINEERING FUNCTION
# ─────────────────────────────────────────────
def apply_feature_engineering(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df["TenureGroup"]  = pd.cut(df["tenure"], bins=[0,12,24,48,float("inf")],
                                 labels=["New","Regular","Established","Loyal"])
    df["IsFirstYear"]  = (df["tenure"] <= 12).astype(int)
    df["IsLongTerm"]   = (df["tenure"] >= 24).astype(int)
    df["AvgMonthlyCharge"] = df.apply(
        lambda x: x["TotalCharges"]/x["tenure"] if x["tenure"]>0 else x["MonthlyCharges"], axis=1)
    df["CustomerLTV"]  = df["TotalCharges"] + (df["MonthlyCharges"] * 6)

    svc = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    df["NumAdditionalServices"] = df[svc].apply(lambda x: (x == "Yes").sum(), axis=1)
    df["HasSecurityBundle"]  = ((df["OnlineSecurity"]=="Yes") & (df["OnlineBackup"]=="Yes")).astype(int)
    df["HasStreamingBundle"] = ((df["StreamingTV"]=="Yes") & (df["StreamingMovies"]=="Yes")).astype(int)
    df["InternetUser"]       = (df["InternetService"] != "No").astype(int)
    df["FiberOpticUser"]     = (df["InternetService"] == "Fiber optic").astype(int)
    df["ServicesPerMonth"]   = df["NumAdditionalServices"] / (df["tenure"] + 1)

    df["IsMonthToMonth"]   = (df["Contract"] == "Month-to-month").astype(int)
    df["ContractType"]     = df["Contract"].map({"Month-to-month":0,"One year":1,"Two year":2})
    df["ElectronicPayment"]= df["PaymentMethod"].str.contains("electronic check|automatic", case=False).astype(int)
    df["PaymentRisk"]      = df["PaymentMethod"].map({
        "Electronic check":3, "Mailed check":2,
        "Bank transfer (automatic)":1, "Credit card (automatic)":1})
    if "PaperlessBilling" in df.columns and df["PaperlessBilling"].dtype == object:
        df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes":1,"No":0})
    df["PaperlessHighRisk"] = ((df["PaperlessBilling"]==1) &
                                (df["PaymentMethod"]=="Electronic check")).astype(int)

    if "Partner" in df.columns and df["Partner"].dtype == object:
        df["Partner"] = df["Partner"].map({"Yes":1,"No":0})
    if "Dependents" in df.columns and df["Dependents"].dtype == object:
        df["Dependents"] = df["Dependents"].map({"Yes":1,"No":0})
    df["HasFamily"] = ((df["Partner"]==1) | (df["Dependents"]==1)).astype(int)

    df["HighCostLowTenure"] = (
        (df["MonthlyCharges"] > df["MonthlyCharges"].median()) & (df["tenure"] < 12)).astype(int)
    df["EngagementScore"] = (
        df["NumAdditionalServices"] * 0.3 +
        df["ContractType"] * 0.4 +
        (df["tenure"] / df["tenure"].max()) * 0.3)
    return df

# ═══════════════════════════════════════════════════════════════════
#  PAGE: HOME & DATA UPLOAD
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Home & Data Upload":
    st.markdown('<div class="main-title">📡 Telecom Churn — End-to-End ML Pipeline</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip([c1,c2,c3,c4], ["4","12","15+","SHAP"],
                              ["Models","Pipeline Steps","Engineered Features","Explainability"]):
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📂 Upload Dataset</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload tele_comm.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df_raw = df
        for k in ["df_feat","models_trained","results_df","tuned_models","shap_vals"]:
            st.session_state[k] = None if k != "models_trained" else False
        st.success(f"✅ Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Preview (first 8 rows)**")
            st.dataframe(df.head(8), use_container_width=True)
        with col2:
            st.markdown("**Column Info**")
            info = pd.DataFrame({
                "dtype" : df.dtypes,
                "nulls" : df.isnull().sum(),
                "unique": df.nunique()
            })
            st.dataframe(info, use_container_width=True)

        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe(include="all"), use_container_width=True)
    else:
        st.info("👆 Upload your `tele_comm.csv` file to start.")
        st.markdown("""
        **Required columns:** `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`,
        `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
        `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,
        `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`
        """)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: EDA
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="main-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    if st.session_state.df_raw is None:
        st.warning("⚠️ Upload a dataset first."); st.stop()

    df = st.session_state.df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target Distribution","📉 Numerics","📦 Categoricals","🔥 Correlation"])

    with tab1:
        counts = df["Churn"].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(["No Churn","Churn"], counts.values, color=["#4C72B0","#DD8452"], edgecolor="black")
        axes[0].set_title("Class Count", fontweight="bold")
        for i, v in enumerate(counts.values):
            axes[0].text(i, v+30, str(v), ha="center", fontweight="bold")
        axes[1].pie(counts.values, labels=["No Churn","Churn"],
                    autopct="%1.1f%%", colors=["#4C72B0","#DD8452"], startangle=90)
        axes[1].set_title("Proportion", fontweight="bold")
        plt.suptitle("Churn Class Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        rate = counts.get(1,0)/len(df)*100
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Customers", f"{len(df):,}")
        c2.metric("Churn Count",     f"{counts.get(1,0):,}")
        c3.metric("Churn Rate",      f"{rate:.1f}%")
        if rate < 35:
            st.warning("⚠️ Imbalanced dataset — SMOTE will be applied during training.")

    with tab2:
        num_col = st.selectbox("Select Feature", ["tenure","MonthlyCharges","TotalCharges"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df[df["Churn"]==0][num_col].hist(ax=axes[0], alpha=0.6, bins=30, label="No Churn", color="#4C72B0")
        df[df["Churn"]==1][num_col].hist(ax=axes[0], alpha=0.6, bins=30, label="Churn",    color="#DD8452")
        axes[0].set_title(f"{num_col} Distribution", fontweight="bold"); axes[0].legend()
        axes[1].boxplot([df[df["Churn"]==0][num_col].dropna(), df[df["Churn"]==1][num_col].dropna()],
                        labels=["No Churn","Churn"], patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", alpha=0.6))
        axes[1].set_title(f"{num_col} Boxplot", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        c1.metric(f"Mean (No Churn)", f"{df[df['Churn']==0][num_col].mean():.2f}")
        c2.metric(f"Mean (Churn)",    f"{df[df['Churn']==1][num_col].mean():.2f}")

    with tab3:
        cat_cols = ["Contract","InternetService","PaymentMethod","TechSupport","OnlineSecurity","gender"]
        available = [c for c in cat_cols if c in df.columns]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        for i, col in enumerate(available[:6]):
            rate_g = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
            rate_g.plot(kind="bar", ax=axes[i], color="#DD8452", edgecolor="black", alpha=0.85)
            axes[i].set_title(f"Churn Rate: {col}", fontweight="bold", fontsize=10)
            axes[i].set_ylim(0, 1)
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=25, ha="right", fontsize=8)
            for bar in axes[i].patches:
                axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                             f"{bar.get_height():.2f}", ha="center", fontsize=8)
        for j in range(len(available), 6): axes[j].set_visible(False)
        plt.suptitle("Churn Rate by Categorical Feature", fontsize=13, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        num_df = df.select_dtypes(include=[np.number])
        corr   = num_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, linewidths=0.4, annot_kws={"size":7}, ax=ax)
        ax.set_title("Correlation Heatmap (lower triangle)", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        if "Churn" in corr:
            top = corr["Churn"].drop("Churn").abs().sort_values(ascending=False).head(6)
            st.markdown("**Top 6 features correlated with Churn:**")
            st.dataframe(top.reset_index().rename(columns={"index":"Feature","Churn":"|Correlation|"}),
                         use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
elif page == "⚙️ Feature Engineering":
    st.markdown('<div class="main-title">⚙️ Feature Engineering</div>', unsafe_allow_html=True)
    if st.session_state.df_raw is None:
        st.warning("⚠️ Upload a dataset first."); st.stop()

    st.markdown("""
    | Group | Engineered Features |
    |---|---|
    | **Tenure** | `TenureGroup` `IsFirstYear` `IsLongTerm` |
    | **Charges** | `AvgMonthlyCharge` `CustomerLTV` |
    | **Services** | `NumAdditionalServices` `HasSecurityBundle` `HasStreamingBundle` `InternetUser` `FiberOpticUser` `ServicesPerMonth` |
    | **Contract/Payment** | `IsMonthToMonth` `ContractType` `ElectronicPayment` `PaymentRisk` `PaperlessHighRisk` |
    | **Demographics** | `HasFamily` |
    | **Risk/Engagement** | `HighCostLowTenure` `EngagementScore` |
    """)

    if st.button("🚀 Apply Feature Engineering", type="primary"):
        with st.spinner("Engineering features..."):
            df_feat = apply_feature_engineering(st.session_state.df_raw)
            st.session_state.df_feat = df_feat
        orig_cols = st.session_state.df_raw.shape[1]
        new_cols  = df_feat.shape[1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Columns",   orig_cols)
        c2.metric("After Engineering",  new_cols)
        c3.metric("Features Added",     new_cols - orig_cols)
        st.success(f"✅ Done! Shape: {df_feat.shape}")
        st.dataframe(df_feat.head(5), use_container_width=True)

        st.markdown('<div class="section-header">New Features vs Churn</div>', unsafe_allow_html=True)
        feats = ["EngagementScore","CustomerLTV","NumAdditionalServices","PaymentRisk"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, feat in zip(axes, feats):
            df_feat[df_feat["Churn"]==0][feat].hist(ax=ax, alpha=0.6, bins=25,
                                                     label="No Churn", color="#4C72B0")
            df_feat[df_feat["Churn"]==1][feat].hist(ax=ax, alpha=0.6, bins=25,
                                                     label="Churn", color="#DD8452")
            ax.set_title(feat, fontweight="bold", fontsize=9); ax.legend(fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    elif st.session_state.df_feat is not None:
        st.info("✅ Feature engineering already applied.")
        st.dataframe(st.session_state.df_feat.head(5), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown('<div class="main-title">🤖 Model Training Pipeline</div>', unsafe_allow_html=True)
    if st.session_state.df_feat is None:
        st.warning("⚠️ Run Feature Engineering first."); st.stop()

    st.markdown('<div class="section-header">⚙️ Pipeline Settings</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size   = st.slider("Test Set Size",            0.10, 0.40, 0.20, 0.05)
        n_splits    = st.slider("CV Folds",                 3, 10, 5)
    with c2:
        fp_cost     = st.number_input("FP Cost (False Alarm $)", value=100, step=50)
        fn_cost     = st.number_input("FN Cost (Missed Churner $)", value=500, step=50)
    with c3:
        n_iter_tune = st.slider("Tuning Iterations / Model", 5, 50, 20)
        smote_on    = st.checkbox("Apply SMOTE", value=True)

    if st.button("🚀 Run Full Training Pipeline", type="primary", use_container_width=True):
        df = st.session_state.df_feat.copy()
        X  = df.drop("Churn", axis=1)
        y  = df["Churn"]

        prog   = st.progress(0)
        status = st.empty()

        # ── Split
        status.info("🔀 Train/Test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        prog.progress(8)

        # ── Encode & Scale
        status.info("🔡 One-Hot Encoding & Scaling...")
        X_tr_enc = pd.get_dummies(X_train, drop_first=True)
        X_te_enc = pd.get_dummies(X_test,  drop_first=True)
        X_tr_enc, X_te_enc = X_tr_enc.align(X_te_enc, join="outer", axis=1, fill_value=0)
        X_tr_enc = X_tr_enc.astype(float)
        X_te_enc = X_te_enc.astype(float)
        feature_names = X_tr_enc.columns.tolist()
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_enc)
        X_te_sc = scaler.transform(X_te_enc)
        prog.progress(18)

        # ── Feature Selection
        status.info("🎯 L1 Feature Selection...")
        l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=1000, random_state=42)
        l1.fit(X_tr_sc, y_train)
        mask = l1.coef_[0] != 0
        sel_feats = np.array(feature_names)[mask]
        X_tr_sel  = X_tr_sc[:, mask]
        X_te_sel  = X_te_sc[:, mask]
        prog.progress(28)

        # ── SMOTE
        status.info("⚖️ Balancing with SMOTE..." if smote_on else "⚖️ Skipping SMOTE...")
        if smote_on:
            X_tr_res, y_tr_res = SMOTE(random_state=42).fit_resample(X_tr_sel, y_train)
        else:
            X_tr_res, y_tr_res = X_tr_sel, y_train
        prog.progress(38)

        # ── Cross-Validation
        status.info("📊 Running 5-fold Cross-Validation on base models...")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        base = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree"      : DecisionTreeClassifier(random_state=42),
            "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost"            : XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                                  random_state=42, n_jobs=-1)
        }
        cv_res = {}
        for name, mdl in base.items():
            scores = {}
            for met in ["roc_auc","f1","recall","precision"]:
                scores[met] = cross_val_score(mdl, X_tr_res, y_tr_res, cv=cv, scoring=met, n_jobs=-1)
            cv_res[name] = scores
        prog.progress(55)

        # ── Hyperparameter Tuning
        status.info("🔧 Hyperparameter tuning (RandomizedSearchCV)...")
        cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        lr_s = RandomizedSearchCV(LogisticRegression(random_state=42),
            {"C":loguniform(0.01,10),"penalty":["l1","l2"],"solver":["liblinear"],"max_iter":[500,1000]},
            n_iter=n_iter_tune, cv=cv3, scoring="roc_auc", n_jobs=-1, random_state=42)
        lr_s.fit(X_tr_res, y_tr_res)

        dt_s = RandomizedSearchCV(DecisionTreeClassifier(random_state=42),
            {"max_depth":randint(3,20),"min_samples_split":randint(2,50),
             "min_samples_leaf":randint(1,30),"criterion":["gini","entropy"]},
            n_iter=n_iter_tune, cv=cv3, scoring="roc_auc", n_jobs=-1, random_state=42)
        dt_s.fit(X_tr_res, y_tr_res)

        rf_s = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators":randint(100,500),"max_depth":[None,10,20,30],
             "min_samples_split":randint(2,20),"min_samples_leaf":randint(1,15),
             "max_features":["sqrt","log2"]},
            n_iter=n_iter_tune, cv=cv3, scoring="roc_auc", n_jobs=-1, random_state=42)
        rf_s.fit(X_tr_res, y_tr_res)

        xgb_s = RandomizedSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1),
            {"n_estimators":randint(100,500),"max_depth":randint(3,10),
             "learning_rate":uniform(0.01,0.3),"subsample":uniform(0.6,0.4),
             "colsample_bytree":uniform(0.6,0.4),"gamma":uniform(0,0.5),
             "reg_alpha":uniform(0,1),"reg_lambda":uniform(1,2)},
            n_iter=n_iter_tune, cv=cv3, scoring="roc_auc", n_jobs=-1, random_state=42)
        xgb_s.fit(X_tr_res, y_tr_res)
        prog.progress(82)

        # ── Evaluate
        status.info("📐 Evaluating on test set...")
        tuned = {"Logistic Regression":lr_s.best_estimator_, "Decision Tree":dt_s.best_estimator_,
                 "Random Forest":rf_s.best_estimator_, "XGBoost":xgb_s.best_estimator_}
        rows, probs = [], {}
        for name, mdl in tuned.items():
            yp = mdl.predict_proba(X_te_sel)[:, 1]
            yd = (yp >= 0.5).astype(int)
            probs[name] = yp
            rows.append({"Model":name,
                         "Accuracy" :round(accuracy_score(y_test,yd),4),
                         "Precision":round(precision_score(y_test,yd),4),
                         "Recall"   :round(recall_score(y_test,yd),4),
                         "F1"       :round(f1_score(y_test,yd),4),
                         "ROC-AUC"  :round(roc_auc_score(y_test,yp),4)})
        res_df = pd.DataFrame(rows).set_index("Model")
        best   = res_df["ROC-AUC"].idxmax()

        # ── Threshold sweep
        yp_best = probs[best]
        trows   = []
        for t in np.arange(0.1, 0.91, 0.05):
            yt = (yp_best >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yt).ravel()
            trows.append({"Threshold":round(t,2),
                          "Precision":round(precision_score(y_test,yt,zero_division=0),4),
                          "Recall"   :round(recall_score(y_test,yt),4),
                          "F1"       :round(f1_score(y_test,yt),4),
                          "BusinessCost": fp*fp_cost + fn*fn_cost,
                          "TN":tn,"FP":fp,"FN":fn,"TP":tp})
        prog.progress(100)

        # Save
        st.session_state.update({
            "models_trained"  : True,
            "results_df"      : res_df,
            "X_test_sel"      : X_te_sel,
            "y_test"          : y_test,
            "tuned_models"    : tuned,
            "selected_features": list(sel_feats),
            "y_prob_dict"     : probs,
            "scaler"          : scaler,
            "l1_mask"         : mask,
            "feature_names"   : feature_names,
            "X_train_res"     : X_tr_res,
            "y_train_res"     : y_tr_res,
            "thresh_df"       : pd.DataFrame(trows),
            "best_model_name" : best,
            "cv_results"      : cv_res,
            "X_test_sel_df"   : pd.DataFrame(X_te_sel, columns=sel_feats),
            "X_train_for_pred": pd.DataFrame(X_tr_res, columns=sel_feats),
            "shap_vals"       : None,
            "explainer"       : None,
        })

        status.success(f"✅ Pipeline complete! Best Model: **{best}** (AUC = {res_df.loc[best,'ROC-AUC']})")
        r = res_df.loc[best]
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Accuracy",  f"{r['Accuracy']:.4f}")
        m2.metric("Precision", f"{r['Precision']:.4f}")
        m3.metric("Recall",    f"{r['Recall']:.4f}")
        m4.metric("F1",        f"{r['F1']:.4f}")
        m5.metric("ROC-AUC",   f"{r['ROC-AUC']:.4f}")

        # CV Results chart
        st.markdown('<div class="section-header">Cross-Validation Results (Base Models)</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        mc = ["#4C72B0","#55A868","#C44E52","#8172B2"]
        for ax, met in zip(axes, ["roc_auc","f1","recall","precision"]):
            means = [cv_res[n][met].mean() for n in base]
            stds  = [cv_res[n][met].std()  for n in base]
            bars  = ax.bar(base.keys(), means, yerr=stds, capsize=4, color=mc, edgecolor="black")
            ax.set_title(f"CV {met.upper()}", fontweight="bold", fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.set_xticklabels(list(base.keys()), rotation=20, ha="right", fontsize=8)
            for bar, m in zip(bars, means):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                        f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Best params
        st.markdown('<div class="section-header">Best Hyperparameters</div>', unsafe_allow_html=True)
        bpc1, bpc2, bpc3, bpc4 = st.columns(4)
        for col, name, search in zip([bpc1,bpc2,bpc3,bpc4],
                                      ["LR","DT","RF","XGB"],
                                      [lr_s,dt_s,rf_s,xgb_s]):
            col.markdown(f"**{name}**")
            col.json(search.best_params_)

    elif st.session_state.models_trained:
        st.info("✅ Models already trained. Navigate to **Model Comparison** to review results.")

# ═══════════════════════════════════════════════════════════════════
#  PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown('<div class="main-title">📈 Model Comparison</div>', unsafe_allow_html=True)
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first."); st.stop()

    res_df   = st.session_state.results_df
    tuned    = st.session_state.tuned_models
    X_te_sel = st.session_state.X_test_sel
    y_test   = st.session_state.y_test
    probs    = st.session_state.y_prob_dict
    best     = st.session_state.best_model_name

    st.markdown(f"### 🏆 Best Model: `{best}`  (ROC-AUC = `{res_df.loc[best,'ROC-AUC']}`)")

    st.markdown('<div class="section-header">Metrics Summary</div>', unsafe_allow_html=True)
    st.dataframe(res_df.style.background_gradient(cmap="YlGn"), use_container_width=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Bar Chart","📉 ROC Curves","📉 PR Curves","🔲 Confusion Matrices","📋 Reports"])

    with tab1:
        fig, ax = plt.subplots(figsize=(13, 5))
        x, w = np.arange(len(res_df)), 0.15
        mc = ["#4C72B0","#55A868","#C44E52","#8172B2","#CCB974"]
        for i, (met, c) in enumerate(zip(["Accuracy","Precision","Recall","F1","ROC-AUC"], mc)):
            bars = ax.bar(x+i*w, res_df[met], w, label=met, color=c, edgecolor="black", alpha=0.85)
        ax.set_xticks(x + w*2)
        ax.set_xticklabels(res_df.index, rotation=15, ha="right")
        ax.set_ylim(0, 1.15); ax.legend(loc="upper right")
        ax.set_title("Model Performance — Test Set", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for (name, mdl), c in zip(tuned.items(), COLORS):
            fpr, tpr, _ = roc_curve(y_test, probs[name])
            ra = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} ({ra:.3f})", color=c, linewidth=2)
        ax.plot([0,1],[0,1],"k--", linewidth=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC-AUC Curves", fontweight="bold"); ax.legend(loc="lower right")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(8, 6))
        for (name, mdl), c in zip(tuned.items(), COLORS):
            prec, rec, _ = precision_recall_curve(y_test, probs[name])
            ax.plot(rec, prec, label=f"{name} ({auc(rec,prec):.3f})", color=c, linewidth=2)
        ax.axhline(y_test.mean(), color="k", linestyle="--", label=f"Baseline ({y_test.mean():.2f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves", fontweight="bold"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, (name, mdl) in zip(axes, tuned.items()):
            cm = confusion_matrix(y_test, mdl.predict(X_te_sel))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"])
            ax.set_title(name, fontweight="bold", fontsize=10)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab5:
        sel = st.selectbox("Select Model", list(tuned.keys()))
        rpt = classification_report(y_test, tuned[sel].predict(X_te_sel),
                                    target_names=["No Churn","Churn"], output_dict=True)
        st.dataframe(pd.DataFrame(rpt).T.round(4), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: THRESHOLD TUNING
# ═══════════════════════════════════════════════════════════════════
elif page == "🎯 Threshold Tuning":
    st.markdown('<div class="main-title">🎯 Threshold Tuning</div>', unsafe_allow_html=True)
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first."); st.stop()

    thresh_df = st.session_state.thresh_df
    best      = st.session_state.best_model_name
    y_test    = st.session_state.y_test
    yp        = st.session_state.y_prob_dict[best]

    st.info(f"💡 Model: **{best}** — Tuning threshold to balance business cost.")

    tab1, tab2, tab3 = st.tabs(["📉 Precision-Recall-F1","💰 Business Cost","📋 Table"])

    with tab1:
        best_f1_t = thresh_df.loc[thresh_df["F1"].idxmax(), "Threshold"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(thresh_df["Threshold"], thresh_df["Precision"], "b-o", label="Precision", markersize=5)
        ax.plot(thresh_df["Threshold"], thresh_df["Recall"],    "r-o", label="Recall",    markersize=5)
        ax.plot(thresh_df["Threshold"], thresh_df["F1"],        "g-o", label="F1 Score",  markersize=5)
        ax.axvline(best_f1_t, color="gray", linestyle="--", label=f"Best F1 @ {best_f1_t:.2f}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_title(f"Threshold Tuning — {best}", fontweight="bold"); ax.legend(); ax.grid(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        c1.metric("Best F1 Threshold",  f"{best_f1_t:.2f}")
        c2.metric("F1 @ Best Threshold",f"{thresh_df['F1'].max():.4f}")

    with tab2:
        best_cost_t = thresh_df.loc[thresh_df["BusinessCost"].idxmin(), "Threshold"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(thresh_df["Threshold"], thresh_df["BusinessCost"], "r-o", markersize=5)
        ax.axvline(best_cost_t, color="green", linestyle="--", label=f"Min Cost @ {best_cost_t:.2f}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Total Business Cost ($)")
        ax.set_title("Business Cost vs Threshold", fontweight="bold"); ax.legend(); ax.grid(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Optimal Threshold",   f"{best_cost_t:.2f}")
        c2.metric("Min Business Cost",   f"${thresh_df['BusinessCost'].min():,.0f}")
        yd_opt = (yp >= best_cost_t).astype(int)
        c3.metric("Recall @ Optimal",    f"{recall_score(y_test, yd_opt):.4f}")

        st.markdown("**Confusion Matrix @ Optimal Threshold**")
        cm = confusion_matrix(y_test, yd_opt)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"])
        ax.set_title(f"threshold = {best_cost_t:.2f}", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        show = thresh_df.drop(columns=["TN","FP","FN","TP"])
        st.dataframe(show.style
                     .background_gradient(cmap="YlGn",   subset=["F1","Recall"])
                     .background_gradient(cmap="YlOrRd_r", subset=["BusinessCost"]),
                     use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Explainability":
    st.markdown('<div class="main-title">🔍 SHAP Explainability</div>', unsafe_allow_html=True)
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first."); st.stop()

    best          = st.session_state.best_model_name
    best_mdl      = st.session_state.tuned_models[best]
    X_te_df       = st.session_state.X_test_sel_df
    X_tr_df       = st.session_state.X_train_for_pred
    yp_best       = st.session_state.y_prob_dict[best]
    sel_feats     = st.session_state.selected_features

    st.markdown(f"Explaining: **{best}** — {len(sel_feats)} selected features")

    if st.button("🔍 Compute SHAP Values", type="primary"):
        with st.spinner("Computing SHAP values..."):
            if best == "Logistic Regression":
                exp = shap.LinearExplainer(best_mdl, X_tr_df)
            else:
                exp = shap.TreeExplainer(best_mdl)
            sv = exp.shap_values(X_te_df)
            sv = sv[1] if isinstance(sv, list) else sv
            st.session_state.shap_vals = sv
            st.session_state.explainer = exp
        st.success("✅ Done!")

    sv  = st.session_state.shap_vals
    exp = st.session_state.explainer
    if sv is None:
        st.info("👆 Click above to compute SHAP values."); st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["🐝 Beeswarm","📊 Bar Plot","💧 Waterfall","📍 Dependence"])

    with tab1:
        fig = plt.figure(figsize=(9, 7))
        shap.summary_plot(sv, X_te_df, plot_type="dot", show=False, max_display=20)
        plt.title(f"SHAP Beeswarm — {best}", fontweight="bold")
        plt.tight_layout(); st.pyplot(plt.gcf()); plt.close("all")

    with tab2:
        fig = plt.figure(figsize=(9, 7))
        shap.summary_plot(sv, X_te_df, plot_type="bar", show=False, max_display=20)
        plt.title(f"SHAP Feature Importance — {best}", fontweight="bold")
        plt.tight_layout(); st.pyplot(plt.gcf()); plt.close("all")

    with tab3:
        opts = {"Highest Churn Risk": int(np.argmax(yp_best)),
                "Lowest Churn Risk" : int(np.argmin(yp_best)),
                "Median Risk"       : int(np.argsort(yp_best)[len(yp_best)//2])}
        sel_lbl = st.selectbox("Customer", list(opts.keys()))
        idx = opts[sel_lbl]
        ev  = exp.expected_value
        if isinstance(ev, list): ev = ev[1]
        expl = shap.Explanation(
            values=sv[idx], base_values=ev,
            data=X_te_df.iloc[idx].values, feature_names=list(sel_feats))
        fig = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(expl, max_display=15, show=False)
        plt.title(f"{sel_lbl}  (P={yp_best[idx]:.3f})", fontweight="bold")
        plt.tight_layout(); st.pyplot(plt.gcf()); plt.close("all")

    with tab4:
        imp      = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(imp)[::-1][:8]
        feat_sel = st.selectbox("Feature", [sel_feats[i] for i in top_idx])
        fi       = list(sel_feats).index(feat_sel)
        fig, ax  = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(X_te_df[feat_sel], sv[:, fi],
                        alpha=0.4, s=12, c=sv[:, fi], cmap="RdYlBu_r")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat_sel); ax.set_ylabel("SHAP Value")
        ax.set_title(f"SHAP Dependence: {feat_sel}", fontweight="bold")
        plt.colorbar(sc, ax=ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════
#  PAGE: SINGLE CUSTOMER PREDICTION
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Single Customer":
    st.markdown('<div class="main-title">🔮 Predict Single Customer</div>', unsafe_allow_html=True)
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first."); st.stop()

    scaler        = st.session_state.scaler
    mask          = st.session_state.l1_mask
    feat_names    = st.session_state.feature_names
    tuned         = st.session_state.tuned_models
    best          = st.session_state.best_model_name
    thresh_df     = st.session_state.thresh_df
    opt_t         = thresh_df.loc[thresh_df["BusinessCost"].idxmin(), "Threshold"]

    st.markdown(f"**Model:** `{best}` &nbsp;|&nbsp; **Optimal Threshold:** `{opt_t:.2f}`")
    st.markdown('<div class="section-header">Customer Details</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        gender     = st.selectbox("Gender",           ["Male","Female"])
        senior     = st.selectbox("Senior Citizen",   ["No","Yes"])
        partner    = st.selectbox("Partner",          ["Yes","No"])
        dependents = st.selectbox("Dependents",       ["No","Yes"])
        tenure     = st.slider("Tenure (months)",     0, 72, 12)
        monthly    = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total      = st.number_input("Total Charges ($)",   0.0, 10000.0, float(tenure * monthly))
    with c2:
        phone_svc  = st.selectbox("Phone Service",    ["Yes","No"])
        multi_line = st.selectbox("Multiple Lines",   ["No","Yes","No phone service"])
        internet   = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        sec        = st.selectbox("Online Security",  ["No","Yes","No internet service"])
        backup     = st.selectbox("Online Backup",    ["No","Yes","No internet service"])
        device     = st.selectbox("Device Protection",["No","Yes","No internet service"])
    with c3:
        tech       = st.selectbox("Tech Support",     ["No","Yes","No internet service"])
        stv        = st.selectbox("Streaming TV",     ["No","Yes","No internet service"])
        smov       = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
        contract   = st.selectbox("Contract",         ["Month-to-month","One year","Two year"])
        paperless  = st.selectbox("Paperless Billing",["Yes","No"])
        payment    = st.selectbox("Payment Method",   ["Electronic check","Mailed check",
                                                        "Bank transfer (automatic)","Credit card (automatic)"])

    if st.button("🔮 Predict Now", type="primary", use_container_width=True):
        raw = pd.DataFrame([{
            "gender":gender, "SeniorCitizen":1 if senior=="Yes" else 0,
            "Partner":partner, "Dependents":dependents, "tenure":tenure,
            "PhoneService":phone_svc, "MultipleLines":multi_line,
            "InternetService":internet, "OnlineSecurity":sec,
            "OnlineBackup":backup, "DeviceProtection":device,
            "TechSupport":tech, "StreamingTV":stv, "StreamingMovies":smov,
            "Contract":contract, "PaperlessBilling":paperless,
            "PaymentMethod":payment, "MonthlyCharges":monthly, "TotalCharges":total, "Churn":0
        }])
        fe  = apply_feature_engineering(raw)
        fe.drop(columns=["Churn"], inplace=True, errors="ignore")
        enc = pd.get_dummies(fe, drop_first=True).reindex(columns=feat_names, fill_value=0).astype(float)
        sc  = scaler.transform(enc)
        sel = sc[:, mask]

        # All model predictions
        st.markdown('<div class="section-header">All Models Prediction</div>', unsafe_allow_html=True)
        prows = []
        for name, mdl in tuned.items():
            prob = mdl.predict_proba(sel)[0, 1]
            pred = "🚨 Will Churn" if prob >= opt_t else "✅ Will Stay"
            prows.append({"Model":name, "Churn Probability":f"{prob:.4f}", "Prediction":pred})
        st.dataframe(pd.DataFrame(prows), use_container_width=True)

        # Best model verdict
        bp    = tuned[best].predict_proba(sel)[0, 1]
        churn = bp >= opt_t

        st.markdown("---")
        if churn:
            st.markdown(f"""
            <div class="risk-high">
            <h3>🚨 HIGH CHURN RISK</h3>
            <b>Model:</b> {best}&nbsp;&nbsp;<b>Probability:</b> {bp:.1%}&nbsp;&nbsp;<b>Threshold:</b> {opt_t:.2f}<br><br>
            <b>💡 Recommended Actions:</b><br>
            &nbsp;• Offer contract upgrade with discount incentive<br>
            &nbsp;• Personalised retention call within 48 hours<br>
            &nbsp;• Review billing plan — consider downgrade option<br>
            &nbsp;• Assign dedicated account manager
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
            <h3>✅ LOW CHURN RISK</h3>
            <b>Model:</b> {best}&nbsp;&nbsp;<b>Probability:</b> {bp:.1%}&nbsp;&nbsp;<b>Threshold:</b> {opt_t:.2f}<br><br>
            <b>💡 Opportunities:</b><br>
            &nbsp;• Cross-sell premium services<br>
            &nbsp;• Enrol in loyalty rewards programme<br>
            &nbsp;• Offer annual plan upgrade
            </div>""", unsafe_allow_html=True)

        # Gauge
        st.markdown("**Churn Probability Gauge**")
        fig, ax = plt.subplots(figsize=(9, 1.8))
        color = "#e74c3c" if churn else "#27ae60"
        ax.barh([""], [bp],        color=color, height=0.5)
        ax.barh([""], [1-bp], left=[bp], color="#ecf0f1", height=0.5)
        ax.axvline(opt_t, color="orange", linestyle="--", linewidth=2,
                   label=f"Threshold = {opt_t:.2f}")
        ax.set_xlim(0, 1)
        ax.text(bp/2, 0, f"{bp:.1%}", ha="center", va="center",
                fontweight="bold", color="white", fontsize=13)
        ax.set_title("Churn Probability", fontweight="bold"); ax.legend(loc="upper right")
        plt.tight_layout(); st.pyplot(fig); plt.close()
