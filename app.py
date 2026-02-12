"""
AUTO-ML SUITE v2.0 — Professional Frontend
Streamlit application with full ML pipeline: ingest → preprocess → train → evaluate → predict
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import base64
import time
import pickle
import warnings
from pathlib import Path

import sys
from pathlib import Path

st.markdown("""
<style>

/* Global background */
.stApp {
    background-color: #0e1117;
    color: #e6e6e6;
}

/* Main content area */
section.main > div {
    background-color: #0e1117;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Cards / containers */
div[data-testid="stVerticalBlock"] {
    background-color: transparent;
}

/* Tables */
thead, tbody, tr, th, td {
    background-color: #0e1117 !important;
    color: #e6e6e6 !important;
}

/* Buttons */
button {
    background-color: #1f2937 !important;
    color: #e5e7eb !important;
    border-radius: 8px;
}

/* Inputs */
input, textarea, select {
    background-color: #111827 !important;
    color: #e5e7eb !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background-color: #111827;
    padding: 10px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


warnings.filterwarnings("ignore")

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto-ML Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Fonts ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Base ──────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #080c14 !important;
    color: #e2e8f0 !important;
}

/* ─── Main container ─────────────────────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem;
    max-width: 1400px;
}

/* ─── Header ────────────────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0d1117 0%, #141b2d 50%, #0d1117 100%);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #f43f5e, #10b981);
}
.app-header-title {
    font-size: 2rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f43f5e 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.app-header-sub {
    color: #64748b; font-size: 0.9rem; font-weight: 400;
}
.badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: #00d4ff;
    font-size: 0.72rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.5px;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d4a !important;
}
[data-testid="stSidebar"] .sidebar-logo {
    font-size: 1.1rem; font-weight: 700; color: #00d4ff;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 16px;
}
[data-testid="stSidebar"] label {
    color: #94a3b8 !important; font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #141b2d !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ─── Cards ──────────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #0d1117 0%, #141b2d 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 18px 20px;
    transition: border-color 0.2s, transform 0.15s;
    height: 100%;
}
.metric-card:hover {
    border-color: #2d4a6e;
    transform: translateY(-2px);
}
.metric-card .m-label {
    font-size: 0.75rem; color: #64748b; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px;
}
.metric-card .m-value {
    font-size: 1.7rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}
.metric-card .m-sub {
    font-size: 0.73rem; color: #475569; margin-top: 4px;
}

/* ─── Section Headings ───────────────────────────────────────────────────── */
.section-heading {
    font-size: 1.15rem; font-weight: 600; color: #e2e8f0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e2d4a;
    margin: 24px 0 16px 0;
    display: flex; align-items: center; gap: 8px;
}
.section-heading .icon {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
}

/* ─── Step indicator ─────────────────────────────────────────────────────── */
.step-bar {
    display: flex; align-items: center; gap: 0;
    background: #0d1117;
    border: 1px solid #1e2d4a;
    border-radius: 12px; padding: 12px 20px;
    margin-bottom: 24px; overflow-x: auto;
}
.step-item {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px; border-radius: 8px;
    font-size: 0.8rem; font-weight: 500; color: #475569;
    white-space: nowrap;
}
.step-item.active {
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #00d4ff;
}
.step-item.done {
    color: #10b981;
}
.step-num {
    width: 20px; height: 20px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    background: #1e2d4a; color: #64748b;
}
.step-item.active .step-num {
    background: rgba(0,212,255,0.2); color: #00d4ff;
}
.step-item.done .step-num {
    background: rgba(16,185,129,0.2); color: #10b981;
}
.step-sep { color: #1e2d4a; padding: 0 4px; font-size: 0.7rem; }

/* ─── Result Table ───────────────────────────────────────────────────────── */
.results-table-container {
    background: #0d1117;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    overflow: hidden;
}
.results-table-container thead {
    background: #141b2d;
}
.results-table-container th {
    color: #64748b !important;
    font-size: 0.76rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.5px;
    padding: 10px 14px !important;
    border-bottom: 1px solid #1e2d4a !important;
}
.results-table-container td {
    font-size: 0.84rem !important;
    padding: 8px 14px !important;
    border-color: #1e2d4a !important;
    color: #cbd5e1 !important;
}

/* ─── Best model banner ──────────────────────────────────────────────────── */
.best-model-banner {
    background: linear-gradient(135deg, rgba(0,212,255,0.05) 0%, rgba(124,58,237,0.05) 100%);
    border: 1px solid rgba(0,212,255,0.25);
    border-left: 4px solid #00d4ff;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.best-model-name {
    font-size: 1.4rem; font-weight: 700; color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}
.best-model-sub {
    color: #64748b; font-size: 0.83rem;
}

/* ─── Alert/Info boxes ───────────────────────────────────────────────────── */
.info-box {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px; padding: 14px 18px;
    color: #94a3b8; font-size: 0.85rem; line-height: 1.6;
    margin: 12px 0;
}
.warn-box {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px; padding: 14px 18px;
    color: #a97326; font-size: 0.85rem; line-height: 1.6;
    margin: 12px 0;
}
.success-box {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 10px; padding: 14px 18px;
    color: #059669; font-size: 0.85rem; line-height: 1.6;
    margin: 12px 0;
}

/* ─── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 20px !important;
    transition: opacity 0.15s, transform 0.1s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] {
    color: #e2e8f0;
}
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 16px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* ─── DataFrames ─────────────────────────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ─── Inputs ─────────────────────────────────────────────────────────────── */
.stSelectbox, .stMultiSelect {
    color: #e2e8f0 !important;
}
input[class*="st-"] {
    background: #141b2d !important;
    border-color: #1e2d4a !important;
    color: #e2e8f0 !important;
}

/* ─── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #7c3aed) !important;
}

/* ─── Expanders ──────────────────────────────────────────────────────────── */
details {
    background: #0d1117 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    padding: 0 !important;
}
summary {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    padding: 12px 16px !important;
    cursor: pointer;
}

/* ─── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d4a6e; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df": None, "target": None, "task_type": "auto",
        "X_processed": None, "y_processed": None,
        "trainer": None, "train_results": None,
        "eval_report": None, "preprocessor": None,
        "preprocessing_log": [],
        "feature_names": None,
        "step": 1,
        "experiments": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
        <div>
            <div class="app-header-title">⚡ Auto-ML Suite</div>
            <div class="app-header-sub">End-to-end automated machine learning · Upload → Preprocess → Train → Evaluate → Predict</div>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <span class="badge">v2.0</span>
            <span class="badge">20+ Algorithms</span>
            <span class="badge">Advanced Evaluation</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Step Bar ────────────────────────────────────────────────────────────────
step = st.session_state.step

def step_class(n):
    if st.session_state.step > n:
        return "done"
    elif st.session_state.step == n:
        return "active"
    return ""

steps = ["Data Ingestion", "Preprocessing", "Feature Engineering", "Model Training", "Evaluation", "Prediction"]
step_icons = ["📁", "🔧", "🔬", "🚀", "📊", "🎯"]
step_html = '<div class="step-bar">'
for i, (name, icon) in enumerate(zip(steps, step_icons), 1):
    cls = step_class(i)
    num_style = ""
    step_html += f"""
    <div class="step-item {cls}">
        <div class="step-num">{i}</div>
        <span>{icon} {name}</span>
    </div>"""
    if i < len(steps):
        step_html += '<span class="step-sep">›</span>'
step_html += '</div>'
st.markdown(step_html, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ Auto-ML Suite</div>', unsafe_allow_html=True)

    st.markdown("**Pipeline Settings**")

    # Task type
    task = st.selectbox("Task Type", ["Auto-Detect", "Classification", "Regression"],
                        help="Auto-detect analyzes the target column")

    st.markdown("---")
    st.markdown("**Preprocessing**")

    missing_strat = st.selectbox(
        "Missing Value Strategy",
        ["auto", "mean", "median", "knn", "iterative", "most_frequent"],
    )
    scaler = st.selectbox(
        "Feature Scaling",
        ["standard", "minmax", "robust", "maxabs", "power", "none"],
    )
    encoding = st.selectbox(
        "Categorical Encoding",
        ["auto", "onehot", "label", "ordinal"],
    )
    outlier = st.selectbox(
        "Outlier Handling",
        ["none", "iqr", "zscore", "winsorize"],
    )
    drop_corr = st.checkbox("Drop High Correlation (≥0.95)", value=False)
    drop_var = st.checkbox("Drop Low Variance Features", value=False)

    st.markdown("---")
    st.markdown("**Training Settings**")

    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    tune_hp = st.checkbox("Hyperparameter Tuning", value=True)
    tuning_method = st.selectbox("Tuning Method", ["random", "grid"])
    n_iter = st.slider("Tuning Iterations", 5, 30, 10)

    st.markdown("---")
    st.markdown("**Feature Engineering**")
    feat_sel = st.checkbox("Feature Selection", value=False)
    feat_sel_method = st.selectbox("Selection Method", ["mutual_info", "f_test", "random_forest"])
    feat_k = st.slider("Top-K Features", 3, 50, 15)
    poly_feat = st.checkbox("Polynomial Features (degree 2)", value=False)
    interaction = st.checkbox("Interaction Features", value=False)

    st.markdown("---")
    st.markdown("**Algorithm Selection**")
    use_all = st.checkbox("Use All Algorithms", value=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab_eda, tab2, tab3, tab4, tab5 = st.tabs([
    "📁  Data",
    "📊  EDA",
    "🔧  Preprocess",
    "🚀  Train",
    "📊  Evaluate",
    "🎯  Predict"
])



# ════════════════════════════════════════════════════════════════════════════
# TAB 1: DATA INGESTION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-heading"><span class="icon">📁</span>Data Ingestion</div>', unsafe_allow_html=True)

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV, Excel (.xlsx, .xls)",
        )

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_ds = st.selectbox("Or load a sample dataset", ["—", "Iris", "Titanic", "Boston Housing", "Wine Quality"])

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.markdown(f'<div class="success-box">✅ Loaded <strong>{uploaded.name}</strong> — {df.shape[0]:,} rows × {df.shape[1]} columns</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

    elif sample_ds != "—":
        try:
            if sample_ds == "Iris":
                from sklearn.datasets import load_iris
                d = load_iris(as_frame=True)
                df = d.frame
                df.columns = list(d.feature_names) + ["target"]
            elif sample_ds == "Titanic":
                df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
            elif sample_ds == "Boston Housing":
                from sklearn.datasets import fetch_california_housing
                d = fetch_california_housing(as_frame=True)
                df = d.frame
            elif sample_ds == "Wine Quality":
                df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/winequality-red.csv", sep=";")
            st.session_state.df = df
            st.markdown(f'<div class="success-box">✅ Loaded sample dataset: <strong>{sample_ds}</strong> — {df.shape[0]:,} rows × {df.shape[1]} columns</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load sample: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df

        # Target selection
        st.markdown('<div class="section-heading"><span class="icon">🎯</span>Target Column</div>', unsafe_allow_html=True)
        target_col = st.selectbox("Select Target (Y) Column", df.columns.tolist())
        st.session_state.target = target_col

        # Overview metrics
        st.markdown('<div class="section-heading"><span class="icon">📊</span>Dataset Overview</div>', unsafe_allow_html=True)
        num_missing = df.isna().sum().sum()
        num_dups = df.duplicated().sum()
        num_num = df.select_dtypes(include=np.number).shape[1]
        num_cat = df.select_dtypes(exclude=np.number).shape[1]

        cols_m = st.columns(6)
        metrics = [
            ("Rows", f"{df.shape[0]:,}", "total samples"),
            ("Columns", str(df.shape[1]), "features + target"),
            ("Numerical", str(num_num), "numeric features"),
            ("Categorical", str(num_cat), "categorical features"),
            ("Missing", f"{num_missing:,}", f"{num_missing/df.size*100:.1f}% of cells"),
            ("Duplicates", f"{num_dups:,}", "duplicate rows"),
        ]
        for col_m, (label, val, sub) in zip(cols_m, metrics):
            with col_m:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="m-label">{label}</div>
                    <div class="m-value">{val}</div>
                    <div class="m-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Data preview
        col_prev, col_types = st.columns([3, 1])
        with col_prev:
            st.markdown("**Preview** (first 100 rows)")
            st.dataframe(df.head(100), use_container_width=True, height=300)
        with col_types:
            st.markdown("**Column Info**")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Missing": df.isna().sum().values,
                "Unique": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True, height=300)

        # Descriptive stats
        with st.expander("📈 Descriptive Statistics"):
            st.dataframe(df.describe(include="all").T.reset_index(), use_container_width=True)

        # Missing value heatmap (if any)
        if num_missing > 0:
            with st.expander("🔍 Missing Values Detail"):
                miss_df = pd.DataFrame({
                    "Column": df.columns,
                    "Missing Count": df.isna().sum().values,
                    "Missing %": (df.isna().mean() * 100).round(2).values,
                }).sort_values("Missing Count", ascending=False)
                miss_df = miss_df[miss_df["Missing Count"] > 0]
                st.dataframe(miss_df, use_container_width=True)

        # Distribution plots
        with st.expander("📊 Column Distributions"):
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use("Agg")

                n = len(num_cols)
                ncols_plot = min(4, n)
                nrows_plot = (n + ncols_plot - 1) // ncols_plot
                fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(ncols_plot * 3.5, nrows_plot * 2.8))
                fig.patch.set_facecolor("#0f1117")
                if n == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                colors = ["#00d4ff", "#7c3aed", "#f43f5e", "#10b981", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6"]
                for i, col in enumerate(num_cols):
                    ax = axes[i]
                    ax.set_facecolor("#1a1f2e")
                    ax.hist(df[col].dropna(), bins=25, color=colors[i % len(colors)], alpha=0.8, edgecolor="none")
                    ax.set_title(col, fontsize=8, color="#94a3b8", pad=4)
                    ax.tick_params(colors="#475569", labelsize=6)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#2d3748")

                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)

                plt.tight_layout(pad=1.0)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0f1117")
                plt.close(fig)
                buf.seek(0)
                st.image(buf, use_container_width=True)

        # Target analysis
        with st.expander(f"🎯 Target Column Analysis: `{target_col}`"):
            y = df[target_col]
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"**Unique Values:** {y.nunique()}")
                st.markdown(f"**Type:** `{y.dtype}`")
                st.markdown(f"**Missing:** {y.isna().sum()}")
            with tc2:
                vc = y.value_counts().head(20)
                st.bar_chart(vc)

        if st.button("✅ Confirm Data & Continue →", use_container_width=True):
            st.session_state.step = max(st.session_state.step, 2)
            st.success("Data confirmed. Head to EDA tab and Proceed for Preprocess Tab.")

    else:
        st.markdown("""
        <div class="info-box">
            📂 <strong>Upload your dataset</strong> to get started, or choose one of the sample datasets above.<br><br>
            Supported formats: CSV, Excel (.xlsx, .xls). The app will auto-detect the ML task type based on your target column.
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: EDA (Exploratory Data Analysis)
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown(
        '<div class="section-heading"><span class="icon">📊</span>Exploratory Data Analysis (EDA)</div>',
        unsafe_allow_html=True
    )

    if st.session_state.df is None:
        st.markdown(
            '<div class="warn-box">⚠️ Please upload data in the <strong>Data</strong> tab first.</div>',
            unsafe_allow_html=True
        )
    else:
        df = st.session_state.df
        target = st.session_state.target

        st.markdown(
            """
            <div class="info-box">
                🔍 EDA helps you understand distributions, relationships, outliers, correlations,
                and the behavior of your target variable <strong>before preprocessing</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )

        # ─── Basic Summary ───────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='m-label'>Rows</div><div class='m-value'>{df.shape[0]:,}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='m-label'>Columns</div><div class='m-value'>{df.shape[1]}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><div class='m-label'>Numerical</div><div class='m-value'>{df.select_dtypes(include=np.number).shape[1]}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card'><div class='m-label'>Categorical</div><div class='m-value'>{df.select_dtypes(exclude=np.number).shape[1]}</div></div>", unsafe_allow_html=True)

        # ─── Missing Values ─────────────────────────────────────────────────
        st.markdown('<div class="section-heading"><span class="icon">❓</span>Missing Values</div>', unsafe_allow_html=True)

        miss = df.isna().mean().mul(100).round(2)
        miss_df = miss[miss > 0].sort_values(ascending=False)

        if miss_df.empty:
            st.success("✅ No missing values found.")
        else:
            st.dataframe(
                miss_df.reset_index().rename(columns={"index": "Column", 0: "Missing %"}),
                use_container_width=True
            )

        # ─── Numerical Distributions ─────────────────────────────────────────
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            st.markdown('<div class="section-heading"><span class="icon">📈</span>Numerical Feature Distributions</div>', unsafe_allow_html=True)

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            sel_num = st.multiselect(
                "Select numerical columns",
                num_cols,
                default=num_cols[:min(6, len(num_cols))]
            )

            if sel_num:
                fig, axes = plt.subplots(len(sel_num), 1, figsize=(6, len(sel_num) * 2.2))
                if len(sel_num) == 1:
                    axes = [axes]

                for ax, col in zip(axes, sel_num):
                    ax.hist(df[col].dropna(), bins=30)
                    ax.set_title(col, fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)

        # ─── Categorical Distributions ───────────────────────────────────────
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if cat_cols:
            st.markdown('<div class="section-heading"><span class="icon">🏷️</span>Categorical Feature Counts</div>', unsafe_allow_html=True)

            sel_cat = st.selectbox("Select categorical column", cat_cols)
            vc = df[sel_cat].value_counts().head(20)
            st.bar_chart(vc)

        # ─── Correlation Matrix ──────────────────────────────────────────────
        if len(num_cols) >= 2:
            st.markdown('<div class="section-heading"><span class="icon">🔗</span>Correlation Matrix</div>', unsafe_allow_html=True)

            corr = df[num_cols].corr()

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(corr)
            ax.set_xticks(range(len(corr)))
            ax.set_yticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticklabels(corr.columns, fontsize=7)
            fig.colorbar(im)
            st.pyplot(fig)

        # ─── Target Analysis ────────────────────────────────────────────────
        if target:
            st.markdown(f'<div class="section-heading"><span class="icon">🎯</span>Target Analysis — {target}</div>', unsafe_allow_html=True)

            if df[target].dtype == "object" or df[target].nunique() < 15:
                st.bar_chart(df[target].value_counts())
            else:
                fig, ax = plt.subplots()
                ax.hist(df[target].dropna(), bins=30)
                st.pyplot(fig)



# ════════════════════════════════════════════════════════════════════════════
# TAB 3: PREPROCESS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="section-heading"><span class="icon">🔧</span>Data Preprocessing & Feature Engineering</div>',
        unsafe_allow_html=True
    )

    if st.session_state.df is None or st.session_state.target is None:
        st.markdown(
            '<div class="warn-box">⚠️ Please upload and confirm your data in the <strong>Data</strong> tab first.</div>',
            unsafe_allow_html=True
        )
    else:
        df = st.session_state.df
        target = st.session_state.target

        st.markdown(
            """
            <div class="info-box">
                ⚙️ Configure your preprocessing pipeline using the sidebar settings, then run it here.
                All transformations are applied to features (X) only — the target column is excluded.
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("▶ Run Preprocessing Pipeline", width="stretch"):
            with st.spinner("Running preprocessing pipeline..."):

                # ✅ Correct local imports (NO modules, NO sys.path hacks)
                from modules.data_preprocessing import DataPreprocessor, FeatureEngineer, DataAnalyzer

                X = df.drop(columns=[target])
                y = df[target]

                preprocessor = DataPreprocessor(
                    missing_strategy=missing_strat,
                    scaling=scaler,
                    encoding=encoding,
                    outlier_method=outlier,
                    drop_high_corr=drop_corr,
                    drop_low_variance=drop_var,
                )

                task_type = {
                    "Auto-Detect": "auto",
                    "Classification": "classification",
                    "Regression": "regression",
                }[task]

                if task_type == "auto":
                    info = DataAnalyzer.profile(df, target)
                    task_type = info["target_info"]["suggested_task"]

                st.session_state.task_type = task_type

                try:
                    X_proc = preprocessor.fit_transform(X, y)
                except Exception as e:
                    st.error(f"Preprocessing error: {e}")
                    st.stop()

                fe = FeatureEngineer()

                # Feature selection
                if feat_sel:
                    k = min(feat_k, X_proc.shape[1])
                    X_proc, selected, sel_info = fe.select_features(
                        X_proc, y, task_type, feat_sel_method, k
                    )
                    st.session_state.preprocessing_log = (
                        preprocessor.preprocessing_log
                        + [f"Feature selection: kept {k} features via {feat_sel_method}"]
                    )
                else:
                    st.session_state.preprocessing_log = preprocessor.preprocessing_log

                # Polynomial features
                if poly_feat:
                    X_proc = fe.generate_polynomial_features(X_proc, degree=2)
                    st.session_state.preprocessing_log.append(
                        f"Polynomial features generated → {X_proc.shape[1]} total"
                    )

                # Interaction features
                if interaction:
                    X_proc = fe.generate_interaction_features(X_proc)
                    st.session_state.preprocessing_log.append(
                        f"Interaction features generated → {X_proc.shape[1]} total"
                    )

                # Clean NaN / inf
                X_proc = X_proc.replace([np.inf, -np.inf], np.nan).fillna(0)

                # Encode target if classification
                from sklearn.preprocessing import LabelEncoder
                if task_type == "classification" and y.dtype == "object":
                    le = LabelEncoder()
                    y = pd.Series(le.fit_transform(y.astype(str)), name=target)

                st.session_state.X_processed = X_proc
                st.session_state.y_processed = y
                st.session_state.preprocessor = preprocessor
                st.session_state.feature_names = X_proc.columns.tolist()
                st.session_state.step = max(st.session_state.step, 3)

                st.markdown(
                    f"""
                    <div class="success-box">
                        ✅ Preprocessing complete!
                        Shape: <strong>{X_proc.shape[0]:,} rows × {X_proc.shape[1]} features</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if st.session_state.X_processed is not None:
            X_proc = st.session_state.X_processed

            p_cols = st.columns(4)
            pm = [
                ("Original Features", str(df.drop(columns=[target]).shape[1])),
                ("After Preprocessing", str(X_proc.shape[1])),
                ("Samples", f"{X_proc.shape[0]:,}"),
                ("Task Type", st.session_state.task_type.title()),
            ]

            for c, (lbl, val) in zip(p_cols, pm):
                with c:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="m-label">{lbl}</div>
                            <div class="m-value" style="font-size:1.2rem;">{val}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if st.session_state.preprocessing_log:
                with st.expander("📋 Preprocessing Log"):
                    for i, entry in enumerate(st.session_state.preprocessing_log, 1):
                        st.markdown(f"`{i}.` {entry}")

            col_xprev, col_yprev = st.columns([3, 1])

            with col_xprev:
                st.markdown("**Processed Features (X)**")
                st.dataframe(X_proc.head(50), width="stretch", height=280)

            with col_yprev:
                st.markdown("**Target (y)**")
                st.dataframe(
                    st.session_state.y_processed.value_counts().reset_index()
                    if st.session_state.task_type == "classification"
                    else st.session_state.y_processed.describe().reset_index(),
                    width="stretch",
                    height=280,
                )

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: TRAIN
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-heading"><span class="icon">🚀</span>Model Training</div>', unsafe_allow_html=True)

    if st.session_state.X_processed is None:
        st.markdown('<div class="warn-box">⚠️ Please complete the <strong>Preprocess</strong> step first.</div>', unsafe_allow_html=True)
    else:
        X_proc = st.session_state.X_processed
        y_proc = st.session_state.y_processed
        task_type = st.session_state.task_type

        st.markdown(f"""
        <div class="info-box">
            🤖 Ready to train on <strong>{X_proc.shape[0]:,} samples</strong> with <strong>{X_proc.shape[1]} features</strong>.
            Task: <strong>{task_type.title()}</strong> | CV: <strong>{cv_folds}-fold</strong> | 
            Tuning: <strong>{"Random Search" if tuning_method == "random" else "Grid Search"}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Algorithm selection
        if not use_all:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from modules.model_training import get_classifier_registry, get_regressor_registry
            algo_reg = get_classifier_registry() if task_type == "classification" else get_regressor_registry()
            algo_names = list(algo_reg.keys())
            selected_algos = st.multiselect("Select Algorithms to Train", algo_names, default=algo_names[:6])
        else:
            selected_algos = None

        if st.button("🚀 Start Training All Models", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from modules.model_training import AutoMLTrainer

            trainer = AutoMLTrainer(
                task_type=task_type,
                test_size=test_size,
                cv_folds=cv_folds,
                tune_hyperparams=tune_hp,
                tuning_method=tuning_method,
                n_iter=n_iter,
                selected_algorithms=selected_algos,
            )

            status_text.text("Initializing training pipeline...")
            progress_bar.progress(10)

            start_time = time.time()
            try:
                results = trainer.train(X_proc, y_proc)
                progress_bar.progress(90)
                total_time = round(time.time() - start_time, 2)
                status_text.text("Finalizing results...")
                time.sleep(0.3)
                progress_bar.progress(100)
                status_text.empty()

                st.session_state.trainer = trainer
                st.session_state.train_results = results
                st.session_state.step = max(st.session_state.step, 4)

                # Best model banner
                best = results["best_model"]
                best_result = next((r for r in results["results"] if r["name"] == best), {})
                primary_metric = "cv_mean" if task_type == "classification" else "r2"
                primary_val = best_result.get(primary_metric, "—")

                st.markdown(f"""
                <div class="best-model-banner">
                    <div style="color:#64748b; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">🏆 Best Model</div>
                    <div class="best-model-name">{best}</div>
                    <div class="best-model-sub">
                        {"CV F1" if task_type == "classification" else "R²"}: <strong style="color:#00d4ff">{primary_val}</strong>
                        &nbsp;·&nbsp; Category: {best_result.get('category', '—')}
                        &nbsp;·&nbsp; Training time: {best_result.get('training_time', '—')}s
                        &nbsp;·&nbsp; Total time: {total_time}s
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Display results table
        if st.session_state.train_results is not None:
            results = st.session_state.train_results
            task_type = st.session_state.task_type

            st.markdown('<div class="section-heading"><span class="icon">📋</span>All Model Results</div>', unsafe_allow_html=True)

            res_list = [r for r in results["results"] if "error" not in r]
            if res_list:
                if task_type == "classification":
                    table_data = pd.DataFrame([{
                        "Model": r["name"],
                        "Category": r.get("category", "—"),
                        "CV Score": r.get("cv_mean", "—"),
                        "CV Std": r.get("cv_std", "—"),
                        "Accuracy": r.get("accuracy", "—"),
                        "F1 (W)": r.get("f1_weighted", "—"),
                        "Precision": r.get("precision", "—"),
                        "Recall": r.get("recall", "—"),
                        "ROC AUC": r.get("roc_auc", "—"),
                        "Train Time (s)": r.get("training_time", "—"),
                    } for r in res_list]).sort_values("CV Score", ascending=False)
                else:
                    table_data = pd.DataFrame([{
                        "Model": r["name"],
                        "Category": r.get("category", "—"),
                        "R²": r.get("r2", "—"),
                        "Adj R²": r.get("r2", "—"),
                        "RMSE": r.get("rmse", "—"),
                        "MAE": r.get("mae", "—"),
                        "MAPE (%)": r.get("mape", "—"),
                        "Expl. Var": r.get("explained_variance", "—"),
                        "CV Score": r.get("cv_mean", "—"),
                        "Train Time (s)": r.get("training_time", "—"),
                    } for r in res_list]).sort_values("R²", ascending=False)

                # Highlight best row
                def highlight_best(row):
                    if row["Model"] == results["best_model"]:
                        return ["background-color: rgba(0,212,255,0.08); color: #00d4ff"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    table_data.style.apply(highlight_best, axis=1),
                    use_container_width=True, height=420
                )

            # Model comparison chart
            st.markdown('<div class="section-heading"><span class="icon">📈</span>Model Comparison Chart</div>', unsafe_allow_html=True)
            try:
                from modules.evaluation import plot_model_comparison
                chart_b64 = plot_model_comparison(res_list, task_type)
                if chart_b64:
                    st.image(base64.b64decode(chart_b64), use_container_width=True)
            except Exception as e:
                st.warning(f"Chart error: {e}")

            # Feature importance
            try:
                trainer = st.session_state.trainer
                fi = trainer.get_feature_importance()
                if fi and st.session_state.feature_names:
                    from modules.evaluation import plot_feature_importance
                    st.markdown('<div class="section-heading"><span class="icon">🔍</span>Feature Importance</div>', unsafe_allow_html=True)
                    fi_b64 = plot_feature_importance(
                        st.session_state.feature_names,
                        fi["values"],
                        f"Feature Importance — {results['best_model']}"
                    )
                    st.image(base64.b64decode(fi_b64), use_container_width=True)
            except Exception:
                pass

            # Best params
            best_res = next((r for r in res_list if r["name"] == results["best_model"]), {})
            if best_res.get("best_params"):
                with st.expander(f"⚙️ Best Hyperparameters — {results['best_model']}"):
                    st.json(best_res["best_params"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 5: EVALUATE
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-heading"><span class="icon">📊</span>In-Depth Model Evaluation</div>', unsafe_allow_html=True)

    if st.session_state.trainer is None:
        st.markdown('<div class="warn-box">⚠️ Please train models in the <strong>Train</strong> tab first.</div>', unsafe_allow_html=True)
    else:
        trainer = st.session_state.trainer
        task_type = st.session_state.task_type
        results = st.session_state.train_results

        if st.button("🔍 Run Full Evaluation Report", use_container_width=True):
            with st.spinner("Generating comprehensive evaluation report..."):
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from modules.evaluation import ClassificationEvaluator, RegressionEvaluator

                    model = trainer.best_model
                    X_train = trainer.X_train
                    X_test = trainer.X_test
                    y_train = trainer.y_train
                    y_test = trainer.y_test
                    feature_names = st.session_state.feature_names

                    if task_type == "classification":
                        ev = ClassificationEvaluator()
                        report = ev.full_report(model, X_test, y_test, X_train, y_train)
                    else:
                        ev = RegressionEvaluator()
                        report = ev.full_report(model, X_test, y_test, X_train, y_train, feature_names)

                    st.session_state.eval_report = report
                    st.session_state.step = max(st.session_state.step, 5)
                    st.success("✅ Evaluation complete!")
                except Exception as e:
                    st.error(f"Evaluation error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.eval_report is not None:
            report = st.session_state.eval_report
            task_type = st.session_state.task_type
            best_name = results["best_model"]

            st.markdown(f"""
            <div class="best-model-banner">
                <div style="color:#64748b; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Evaluating</div>
                <div class="best-model-name">{best_name}</div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            if task_type == "classification":
                metric_items = [
                    ("Accuracy", report.get("accuracy")),
                    ("F1 Weighted", report.get("f1_weighted")),
                    ("Precision", report.get("precision_weighted")),
                    ("Recall", report.get("recall_weighted")),
                    ("ROC AUC", report.get("roc_auc", report.get("roc_auc_ovr", "—"))),
                    ("MCC", report.get("mcc")),
                    ("Cohen Kappa", report.get("cohen_kappa")),
                    ("Log Loss", report.get("log_loss", "—")),
                ]
            else:
                metric_items = [
                    ("R²", report.get("r2")),
                    ("Adj. R²", report.get("adjusted_r2")),
                    ("RMSE", report.get("rmse")),
                    ("MAE", report.get("mae")),
                    ("MSE", report.get("mse")),
                    ("MAPE (%)", report.get("mape", "—")),
                    ("Expl. Var.", report.get("explained_variance")),
                    ("Resid. Std", report.get("residuals_std")),
                ]

            mi_cols = st.columns(4)
            for i, (lbl, val) in enumerate(metric_items[:8]):
                with mi_cols[i % 4]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="m-label">{lbl}</div>
                        <div class="m-value">{val if val is not None else '—'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                if (i + 1) % 4 == 0 and i < 7:
                    st.markdown("<br>", unsafe_allow_html=True)
                    mi_cols = st.columns(4)

            st.markdown("<br>", unsafe_allow_html=True)

            # Plots
            plots = report.get("plots", {})
            plot_tabs_labels = [k.replace("_", " ").title() for k in plots.keys() if plots[k]]
            plot_keys = [k for k, v in plots.items() if v]

            if plot_keys:
                st.markdown('<div class="section-heading"><span class="icon">📈</span>Evaluation Plots</div>', unsafe_allow_html=True)
                plot_tab_list = st.tabs(plot_tabs_labels)
                for ptab, pk in zip(plot_tab_list, plot_keys):
                    with ptab:
                        img_data = plots[pk]
                        if img_data:
                            st.image(base64.b64decode(img_data), use_container_width=True)

            # Confusion matrix details (classification)
            if task_type == "classification" and "confusion_matrix" in report:
                with st.expander("🔢 Confusion Matrix Raw Values"):
                    cm_df = pd.DataFrame(report["confusion_matrix"])
                    st.dataframe(cm_df, use_container_width=True)

            # Per-class report (classification)
            if task_type == "classification":
                with st.expander("📋 Full Classification Report"):
                    from sklearn.metrics import classification_report
                    y_pred = trainer.best_model.predict(trainer.X_test)
                    st.text(classification_report(trainer.y_test, y_pred, zero_division=0))

            # Permutation importance (regression)
            if "permutation_importance" in report:
                with st.expander("🔍 Permutation Importance Details"):
                    perm = report["permutation_importance"]
                    perm_df = pd.DataFrame({
                        "Feature": perm["features"],
                        "Importance Mean": [round(x, 6) for x in perm["importances_mean"]],
                        "Importance Std": [round(x, 6) for x in perm["importances_std"]],
                    }).sort_values("Importance Mean", ascending=False)
                    st.dataframe(perm_df, use_container_width=True)

            # Download model
            st.markdown('<div class="section-heading"><span class="icon">💾</span>Export</div>', unsafe_allow_html=True)
            dc1, dc2 = st.columns(2)
            with dc1:
                model_bytes = pickle.dumps({
                    "model": trainer.best_model,
                    "name": trainer.best_model_name,
                    "task_type": task_type,
                    "features": st.session_state.feature_names,
                })
                st.download_button(
                    "⬇️ Download Best Model (.pkl)",
                    data=model_bytes,
                    file_name=f"automl_{trainer.best_model_name.replace(' ', '_').lower()}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            with dc2:
                # Export results CSV
                if st.session_state.train_results:
                    res_df = pd.DataFrame([
                        {k: v for k, v in r.items() if k not in ["confusion_matrix", "classification_report"]}
                        for r in st.session_state.train_results["results"]
                    ])
                    csv_bytes = res_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Results (.csv)",
                        data=csv_bytes,
                        file_name="automl_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6: PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-heading"><span class="icon">🎯</span>Make Predictions</div>', unsafe_allow_html=True)

    if st.session_state.trainer is None:
        st.markdown('<div class="warn-box">⚠️ Please train models in the <strong>Train</strong> tab first.</div>', unsafe_allow_html=True)
    else:
        trainer = st.session_state.trainer
        feature_names = st.session_state.feature_names or []
        task_type = st.session_state.task_type

        pred_mode = st.radio(
            "Prediction Mode",
            ["Manual Input", "Upload Prediction File"],
            horizontal=True,
        )

        if pred_mode == "Manual Input":
            st.markdown("**Enter feature values manually:**")
            st.markdown(f'<div class="info-box">Provide values for all <strong>{len(feature_names)}</strong> features used during training.</div>', unsafe_allow_html=True)

            n_feat_cols = min(4, len(feature_names))
            input_cols = st.columns(n_feat_cols)
            input_vals = {}
            for i, fname in enumerate(feature_names):
                with input_cols[i % n_feat_cols]:
                    input_vals[fname] = st.number_input(fname, value=0.0, format="%.4f", key=f"inp_{i}")

            if st.button("🎯 Predict", use_container_width=True):
                try:
                    X_input = pd.DataFrame([input_vals])
                    pred = trainer.predict(X_input)
                    prediction = pred[0]

                    st.markdown(f"""
                    <div class="best-model-banner">
                        <div style="color:#64748b; font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">
                            Prediction from {trainer.best_model_name}
                        </div>
                        <div class="best-model-name">{prediction}</div>
                        <div class="best-model-sub">
                            Task: {task_type.title()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probabilities for classification
                    if task_type == "classification":
                        try:
                            proba = trainer.predict_proba(X_input)[0]
                            classes = trainer.best_model.classes_
                            proba_df = pd.DataFrame({
                                "Class": classes,
                                "Probability": [round(float(p), 4) for p in proba],
                            }).sort_values("Probability", ascending=False)
                            st.markdown("**Class Probabilities:**")
                            st.dataframe(proba_df, use_container_width=True)
                        except Exception:
                            pass
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        else:
            pred_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
            if pred_file:
                try:
                    pred_df = pd.read_csv(pred_file)
                    st.markdown(f"Loaded {pred_df.shape[0]:,} rows for prediction.")

                    # Auto-align columns
                    available_feats = [f for f in feature_names if f in pred_df.columns]
                    missing_feats = [f for f in feature_names if f not in pred_df.columns]

                    if missing_feats:
                        st.markdown(f'<div class="warn-box">⚠️ Missing features (will be filled with 0): {", ".join(missing_feats[:10])}</div>', unsafe_allow_html=True)
                        for f in missing_feats:
                            pred_df[f] = 0.0

                    X_pred = pred_df[feature_names].fillna(0)

                    if st.button("🎯 Run Batch Predictions", use_container_width=True):
                        preds = trainer.predict(X_pred)
                        pred_df["__prediction__"] = preds

                        if task_type == "classification":
                            try:
                                probas = trainer.predict_proba(X_pred)
                                classes = trainer.best_model.classes_
                                for i, cls in enumerate(classes):
                                    pred_df[f"__prob_{cls}__"] = probas[:, i].round(4)
                            except Exception:
                                pass

                        st.dataframe(pred_df.head(200), use_container_width=True, height=350)

                        csv_out = pred_df.to_csv(index=False).encode()
                        st.download_button(
                            "⬇️ Download Predictions CSV",
                            data=csv_out,
                            file_name="predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                        st.markdown(f'<div class="success-box">✅ Generated {len(preds):,} predictions successfully!</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 16px 0; border-top: 1px solid #1e2d4a; margin-top: 40px;">
    <div style="color: #1e2d4a; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;">
        ⚡ Auto-ML Suite v2.0 &nbsp;·&nbsp; 20+ Algorithms &nbsp;·&nbsp; Built with Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

