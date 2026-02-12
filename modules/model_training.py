"""
Model Training Module — Auto-ML Suite v2.0
Supports 20+ algorithms for classification and regression with full hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import json
import os
import time
import warnings
from typing import Optional, Dict, Any, List, Tuple

warnings.filterwarnings("ignore")

# ─── Sklearn Core ────────────────────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline

# ─── Classification Algorithms ────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier,
    VotingClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# ─── Regression Algorithms ────────────────────────────────────────────────────
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, Lars, SGDRegressor, PassiveAggressiveRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# ─── Optional Boosting Libraries ─────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

import pickle
import logging

logger = logging.getLogger(__name__)

# ─── Algorithm Registries ─────────────────────────────────────────────────────

def get_classifier_registry():
    registry = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "saga"]},
            "category": "Linear",
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
            "category": "Tree",
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "category": "Ensemble",
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
            "category": "Ensemble",
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
            "category": "Ensemble",
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [None, 10]},
            "category": "Ensemble",
        },
        "Hist Gradient Boosting": {
            "model": HistGradientBoostingClassifier(random_state=42),
            "params": {"learning_rate": [0.05, 0.1], "max_iter": [100, 200]},
            "category": "Ensemble",
        },
        "SVM (RBF)": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
            "category": "SVM",
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7, 11], "metric": ["euclidean", "manhattan"]},
            "category": "Instance-Based",
        },
        "Gaussian Naive Bayes": {
            "model": GaussianNB(),
            "params": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
            "category": "Probabilistic",
        },
        "LDA": {
            "model": LinearDiscriminantAnalysis(),
            "params": {"solver": ["svd", "lsqr"]},
            "category": "Linear",
        },
        "MLP Neural Network": {
            "model": MLPClassifier(max_iter=500, random_state=42),
            "params": {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "activation": ["relu", "tanh"]},
            "category": "Neural Network",
        },
        "Ridge Classifier": {
            "model": RidgeClassifier(),
            "params": {"alpha": [0.1, 1.0, 10.0]},
            "category": "Linear",
        },
        "SGD Classifier": {
            "model": SGDClassifier(random_state=42, max_iter=1000),
            "params": {"alpha": [0.0001, 0.001], "loss": ["hinge", "log_loss"]},
            "category": "Linear",
        },
    }
    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = {
            "model": XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6]},
            "category": "Boosting",
        }
    if LIGHTGBM_AVAILABLE:
        registry["LightGBM"] = {
            "model": LGBMClassifier(random_state=42, verbose=-1),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
            "category": "Boosting",
        }
    if CATBOOST_AVAILABLE:
        registry["CatBoost"] = {
            "model": CatBoostClassifier(random_state=42, verbose=0),
            "params": {"iterations": [100, 200], "learning_rate": [0.05, 0.1]},
            "category": "Boosting",
        }
    return registry


def get_regressor_registry():
    registry = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {"fit_intercept": [True, False]},
            "category": "Linear",
        },
        "Ridge Regression": {
            "model": Ridge(),
            "params": {"alpha": [0.1, 1.0, 10.0, 100.0]},
            "category": "Linear",
        },
        "Lasso Regression": {
            "model": Lasso(max_iter=5000),
            "params": {"alpha": [0.01, 0.1, 1.0, 10.0]},
            "category": "Linear",
        },
        "Elastic Net": {
            "model": ElasticNet(max_iter=5000),
            "params": {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
            "category": "Linear",
        },
        "Bayesian Ridge": {
            "model": BayesianRidge(),
            "params": {"alpha_1": [1e-6, 1e-5], "lambda_1": [1e-6, 1e-5]},
            "category": "Bayesian",
        },
        "Huber Regressor": {
            "model": HuberRegressor(max_iter=500),
            "params": {"epsilon": [1.1, 1.35, 1.5], "alpha": [0.0001, 0.001]},
            "category": "Robust",
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5]},
            "category": "Tree",
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
            "category": "Ensemble",
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
            "category": "Ensemble",
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
            "category": "Ensemble",
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(random_state=42),
            "params": {"n_estimators": [50, 100]},
            "category": "Ensemble",
        },
        "Hist Gradient Boosting": {
            "model": HistGradientBoostingRegressor(random_state=42),
            "params": {"learning_rate": [0.05, 0.1], "max_iter": [100, 200]},
            "category": "Ensemble",
        },
        "SVR": {
            "model": SVR(),
            "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
            "category": "SVM",
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsRegressor(),
            "params": {"n_neighbors": [3, 5, 7, 11]},
            "category": "Instance-Based",
        },
        "MLP Neural Network": {
            "model": MLPRegressor(max_iter=500, random_state=42),
            "params": {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "activation": ["relu", "tanh"]},
            "category": "Neural Network",
        },
        "SGD Regressor": {
            "model": SGDRegressor(max_iter=1000, random_state=42),
            "params": {"alpha": [0.0001, 0.001], "loss": ["squared_error", "huber"]},
            "category": "Linear",
        },
    }
    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = {
            "model": XGBRegressor(random_state=42, verbosity=0),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6]},
            "category": "Boosting",
        }
    if LIGHTGBM_AVAILABLE:
        registry["LightGBM"] = {
            "model": LGBMRegressor(random_state=42, verbose=-1),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
            "category": "Boosting",
        }
    if CATBOOST_AVAILABLE:
        registry["CatBoost"] = {
            "model": CatBoostRegressor(random_state=42, verbose=0),
            "params": {"iterations": [100, 200], "learning_rate": [0.05, 0.1]},
            "category": "Boosting",
        }
    return registry


# ─── Core Training Engine ──────────────────────────────────────────────────────

class AutoMLTrainer:
    def __init__(self, task_type: str = "auto", test_size: float = 0.2,
                 cv_folds: int = 5, tune_hyperparams: bool = True,
                 tuning_method: str = "random", n_iter: int = 10,
                 scoring: Optional[str] = None, selected_algorithms: Optional[List[str]] = None):
        self.task_type = task_type
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.tune_hyperparams = tune_hyperparams
        self.tuning_method = tuning_method
        self.n_iter = n_iter
        self.scoring = scoring
        self.selected_algorithms = selected_algorithms
        self.results: List[Dict] = []
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _detect_task(self, y: pd.Series) -> str:
        n_unique = y.nunique()
        if y.dtype in ["object", "category", "bool"] or n_unique <= 20:
            return "classification"
        return "regression"

    def _get_scoring(self) -> str:
        if self.scoring:
            return self.scoring
        if self.task_type == "classification":
            return "f1_weighted"
        return "r2"

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        # Auto-detect task
        if self.task_type == "auto":
            self.task_type = self._detect_task(y)
        logger.info(f"Task type detected: {self.task_type}")

        # Split
        if self.task_type == "classification":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )

        # Registry
        registry = (
            get_classifier_registry()
            if self.task_type == "classification"
            else get_regressor_registry()
        )

        if self.selected_algorithms:
            registry = {k: v for k, v in registry.items() if k in self.selected_algorithms}

        scoring = self._get_scoring()
        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            if self.task_type == "classification"
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        )

        self.results = []
        for name, config in registry.items():
            logger.info(f"Training: {name}")
            start = time.time()
            try:
                model = config["model"]
                params = config["params"]

                if self.tune_hyperparams and params:
                    if self.tuning_method == "grid":
                        searcher = GridSearchCV(model, params, cv=cv, scoring=scoring, n_jobs=-1)
                    else:
                        n = min(self.n_iter, np.prod([len(v) for v in params.values()]))
                        searcher = RandomizedSearchCV(
                            model, params, n_iter=int(n), cv=cv,
                            scoring=scoring, n_jobs=-1, random_state=42
                        )
                    searcher.fit(self.X_train, self.y_train)
                    best = searcher.best_estimator_
                    best_params = searcher.best_params_
                else:
                    best = model
                    best.fit(self.X_train, self.y_train)
                    best_params = {}

                # Cross-val
                cv_scores = cross_val_score(best, self.X_train, self.y_train, cv=cv, scoring=scoring)
                y_pred = best.predict(self.X_test)
                elapsed = round(time.time() - start, 3)

                if self.task_type == "classification":
                    metrics = self._classification_metrics(best, y_pred)
                else:
                    metrics = self._regression_metrics(y_pred)

                result = {
                    "name": name,
                    "category": config["category"],
                    "cv_mean": round(float(cv_scores.mean()), 4),
                    "cv_std": round(float(cv_scores.std()), 4),
                    "cv_scores": [round(float(s), 4) for s in cv_scores],
                    "best_params": best_params,
                    "training_time": elapsed,
                    **metrics,
                }
                self.results.append({"model": best, **result})

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                self.results.append({"name": name, "error": str(e), "cv_mean": -999})

        # Sort results
        sort_key = "cv_mean" if self.task_type == "classification" else "r2"
        if self.task_type == "regression":
            valid = [r for r in self.results if "r2" in r]
            valid.sort(key=lambda x: x.get("r2", -999), reverse=True)
        else:
            valid = [r for r in self.results if "cv_mean" in r and r["cv_mean"] != -999]
            valid.sort(key=lambda x: x["cv_mean"], reverse=True)

        if valid:
            self.best_model = valid[0]["model"]
            self.best_model_name = valid[0]["name"]

        # Build summary (exclude model objects)
        summary = []
        for r in self.results:
            entry = {k: v for k, v in r.items() if k != "model"}
            summary.append(entry)

        return {
            "task_type": self.task_type,
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "n_features": self.X_train.shape[1],
            "n_algorithms": len(registry),
            "best_model": self.best_model_name,
            "results": summary,
        }

    def _classification_metrics(self, model, y_pred) -> Dict:
        m = {
            "accuracy": round(float(accuracy_score(self.y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(self.y_test, y_pred, average="weighted", zero_division=0)), 4),
            "precision": round(float(precision_score(self.y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(self.y_test, y_pred, average="weighted", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
            "classification_report": classification_report(self.y_test, y_pred, output_dict=True, zero_division=0),
        }
        # ROC AUC
        try:
            if hasattr(model, "predict_proba"):
                classes = np.unique(self.y_train)
                if len(classes) == 2:
                    proba = model.predict_proba(self.X_test)[:, 1]
                    m["roc_auc"] = round(float(roc_auc_score(self.y_test, proba)), 4)
                else:
                    proba = model.predict_proba(self.X_test)
                    m["roc_auc"] = round(float(roc_auc_score(self.y_test, proba, multi_class="ovr", average="weighted")), 4)
        except Exception:
            pass
        return m

    def _regression_metrics(self, y_pred) -> Dict:
        mse = mean_squared_error(self.y_test, y_pred)
        return {
            "r2": round(float(r2_score(self.y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "mae": round(float(mean_absolute_error(self.y_test, y_pred)), 4),
            "mse": round(float(mse), 4),
            "mape": round(float(mean_absolute_percentage_error(self.y_test, y_pred) * 100), 4),
            "explained_variance": round(float(explained_variance_score(self.y_test, y_pred)), 4),
        }

    def save_model(self, path: str = "models/best_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.best_model, "name": self.best_model_name, "task_type": self.task_type}, f)
        return path

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("No model trained yet.")
        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("No model trained yet.")
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X)
        raise ValueError("Model does not support probability predictions.")

    def get_feature_importance(self) -> Optional[Dict]:
        if self.best_model is None:
            return None
        if hasattr(self.best_model, "feature_importances_"):
            fi = self.best_model.feature_importances_
            return {"type": "feature_importances", "values": fi.tolist()}
        if hasattr(self.best_model, "coef_"):
            coef = self.best_model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            return {"type": "coefficients", "values": coef.tolist()}
        return None
