# ⚡ Auto-ML Suite v2.0

> **Production-grade AutoML platform** — Upload data, configure a pipeline, and automatically benchmark 20+ algorithms across classification and regression tasks with EDA report,deep evaluation, feature engineering, and batch prediction support.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square)
![Algorithms](https://img.shields.io/badge/Algorithms-20%2B-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 🚀 What's New in v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Algorithms | ~6 | **20+ (Classifier + Regressor)** |
|Exploratory Data Analysis | Simple | **Graphs, Bar Charts, Confusion MAtrix**|
| Hyperparameter Tuning | Basic | **Random + Grid Search** |
| Evaluation Metrics | Basic | **12+ metrics + plots** |
| Feature Engineering | None | **Selection, PCA, Polynomial, Interactions** |
| Visualization | Minimal | **ROC, PR, Calibration, Learning Curves, Residuals** |
| Outlier Handling | None | **IQR, Z-score, Winsorize** |
| Missing Values | Simple | **Mean/Median/Mode, KNN, Iterative** |
| API | Basic | **Extended with Evaluation + Feature + Experiments routers** |
| UI | Plain | **Professional dark theme with full pipeline UX** |
| Export | None | **Model .pkl + Results .csv download** |
| Batch Prediction | None | **Upload CSV → Download predictions** |


---

## 🧬 Supported Algorithms

### Classification (16+)
| Algorithm | Category |
|-----------|----------|
| Logistic Regression | Linear |
| Ridge Classifier | Linear |
| SGD Classifier | Linear |
| Linear Discriminant Analysis | Linear |
| Decision Tree | Tree |
| Random Forest | Ensemble |
| Extra Trees | Ensemble |
| Gradient Boosting | Ensemble |
| Hist Gradient Boosting | Ensemble |
| AdaBoost | Ensemble |
| SVM (RBF kernel) | SVM |
| K-Nearest Neighbors | Instance-Based |
| Gaussian Naive Bayes | Probabilistic |
| MLP Neural Network | Neural Network |
| **XGBoost** *(optional)* | Boosting |
| **LightGBM** *(optional)* | Boosting |
| **CatBoost** *(optional)* | Boosting |

### Regression (17+)
| Algorithm | Category |
|-----------|----------|
| Linear / Ridge / Lasso / Elastic Net | Linear |
| Bayesian Ridge | Bayesian |
| Huber Regressor | Robust |
| Decision Tree | Tree |
| Random Forest | Ensemble |
| Extra Trees | Ensemble |
| Gradient Boosting | Ensemble |
| Hist Gradient Boosting | Ensemble |
| AdaBoost | Ensemble |
| SVR | SVM |
| K-Nearest Neighbors | Instance-Based |
| MLP Neural Network | Neural Network |
| SGD Regressor | Linear |
| **XGBoost** *(optional)* | Boosting |
| **LightGBM** *(optional)* | Boosting |
| **CatBoost** *(optional)* | Boosting |

---

## 📊 Evaluation Metrics

### Classification
- Accuracy, F1 (Weighted + Macro), Precision, Recall
- ROC AUC, Average Precision, Log Loss
- Matthews Correlation Coefficient (MCC), Cohen's Kappa
- Brier Score, Confusion Matrix
- **Plots:** ROC Curve, Precision-Recall Curve, Calibration Curve, Confusion Matrix Heatmap, Learning Curve

### Regression
- R², Adjusted R², RMSE, MAE, MSE, MAPE
- Explained Variance, Residual Mean + Std
- **Plots:** Actual vs Predicted, Residuals, Residuals Distribution, Error by Quantile, Learning Curve, Permutation Importance

---

## 🏗️ Project Structure
┌──────────────────┐
│   User Interface │
│   (Streamlit UI) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   app.py         │
│  (Main Controller)│
└────────┬─────────┘
         │
         ▼
┌───────────────────────────────┐
│        Router Layer            │
│  ┌─────────────────────────┐ │
│  │ data_router.py           │ │
│  │ feature_router.py        │ │
│  │ model_router.py          │ │
│  │ experiment_router.py     │ │
│  │ evaluation_router.py     │ │
│  └─────────────────────────┘ │
└────────┬──────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│            Core ML Modules               │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ data_preprocessing.py              │  │
│  │ - Data cleaning                    │  │
│  │ - Encoding & scaling               │  │
│  │ - Feature engineering              │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ model_training.py                  │  │
│  │ - Train ML models                  │  │
│  │ - Model selection                  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ evaluation.py                      │  │
│  │ - Metrics & performance analysis   │  │
│  │ - Confusion matrix / ROC-AUC       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AUTO-ML-SUITE.git
cd AUTO-ML-SUITE
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt

# Optional: install boosting libraries
pip install xgboost lightgbm catboost
```

### 4. Run the Streamlit frontend
```bash
streamlit run frontend/app.py
```

### 5. Run the FastAPI backend *(optional — UI is standalone)*
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
API docs at: `http://localhost:8000/api/docs`

---

## 🖥️ Streamlit UI Walkthrough

### Tab 1 — 📁 Data
- Upload CSV / Excel or choose a built-in sample dataset
- Select your target (Y) column
- View dataset profile: shape, dtypes, missing values, duplicates
- Explore column distributions and target analysis

### Tab 2 — 🔧 Preprocess
Configure via the sidebar, then click **Run Preprocessing Pipeline**:
- Missing value imputation (Mean, Median, KNN, Iterative)
- Outlier handling (IQR clipping, Z-score, Winsorization)
- Categorical encoding (One-Hot, Label, Ordinal)
- Feature scaling (Standard, MinMax, Robust, Power)
- Feature selection (Mutual Info, F-test, Random Forest importance)
- Polynomial feature generation (degree 2)
- Interaction feature generation

### Tab 3 — 🚀 Train
Click **Start Training All Models** to benchmark all algorithms:
- Auto-detects classification vs. regression from target column
- Runs k-fold cross-validation on all models
- Optional: RandomizedSearchCV or GridSearchCV for hyperparameter tuning
- Displays ranked results table with all metrics
- Shows model comparison chart and feature importance

### Tab 4 — 📊 Evaluate
Click **Run Full Evaluation Report** for deep analysis:
- 8+ computed metrics in a clear dashboard
- Interactive plot viewer: ROC, PR, calibration, learning curves, etc.
- Downloadable: best model `.pkl` + results `.csv`

### Tab 5 — 🎯 Predict
Two prediction modes:
- **Manual Input:** enter feature values one by one, get instant prediction + class probabilities
- **Batch Upload:** upload a CSV → get predictions appended → download results

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/data/upload` | Upload CSV/Excel dataset |
| GET | `/api/data/columns` | List columns and dtypes |
| POST | `/api/data/preprocess` | Run preprocessing pipeline |
| POST | `/api/data/correlation` | Get correlation matrix |
| POST | `/api/model/train` | Train all algorithms |
| POST | `/api/model/predict` | Predict with best model |
| GET | `/api/model/algorithms` | List available algorithms |
| GET | `/api/model/model/info` | Best model info + feature importance |
| POST | `/api/evaluation/report` | Full evaluation report |
| GET | `/api/evaluation/metrics` | Quick metric summary |
| POST | `/api/features/select` | Feature selection |
| POST | `/api/features/pca` | PCA dimensionality reduction |
| POST | `/api/experiments/log` | Log an experiment run |
| GET | `/api/experiments/list` | List all experiments |
| GET | `/api/experiments/compare/models` | Compare experiment runs |

---

## 🔮 Roadmap

- [ ] SHAP explainability integration
- [ ] Time-series forecasting support
- [ ] MLflow/Weights & Biases experiment tracking integration
- [ ] Multi-label classification support
- [ ] Custom metric definitions
- [ ] Data drift detection (evidently)
- [ ] Auto-ensembling / stacking
- [ ] Model versioning with DVC
- [ ] Docker Compose deployment config

---

## 📄 License

MIT © 2024 Auto-ML Suite Contributors
