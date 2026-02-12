"""
Evaluation Module — Auto-ML Suite v2.0
Advanced metrics, learning curves, calibration, SHAP analysis, and drift detection.
"""

import numpy as np
import pandas as pd
import io
import base64
import logging
from typing import Dict, List, Optional, Any

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    log_loss, matthews_corrcoef, cohen_kappa_score, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.inspection import permutation_importance, partial_dependence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns

logger = logging.getLogger(__name__)

# ─── Plotting Helpers ─────────────────────────────────────────────────────────

PALETTE = ["#00d4ff", "#7c3aed", "#f43f5e", "#10b981", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6"]

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f1117")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def _style_fig(fig, ax=None):
    fig.patch.set_facecolor("#0f1117")
    if ax:
        axes = [ax] if not isinstance(ax, list) else ax
        for a in axes:
            a.set_facecolor("#1a1f2e")
            a.tick_params(colors="#94a3b8", labelsize=9)
            a.xaxis.label.set_color("#94a3b8")
            a.yaxis.label.set_color("#94a3b8")
            a.title.set_color("#e2e8f0")
            for spine in a.spines.values():
                spine.set_edgecolor("#2d3748")


# ─── Classification Evaluator ─────────────────────────────────────────────────

class ClassificationEvaluator:

    def full_report(self, model, X_test, y_test, X_train=None, y_train=None) -> Dict:
        y_pred = model.predict(X_test)
        report = {}

        # Core metrics
        report["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
        report["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        report["f1_macro"] = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
        report["precision_weighted"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        report["recall_weighted"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        report["mcc"] = round(float(matthews_corrcoef(y_test, y_pred)), 4)
        report["cohen_kappa"] = round(float(cohen_kappa_score(y_test, y_pred)), 4)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        report["confusion_matrix"] = cm.tolist()

        # Probabilistic metrics
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            classes = model.classes_
            if len(classes) == 2:
                report["roc_auc"] = round(float(roc_auc_score(y_test, proba[:, 1])), 4)
                report["log_loss"] = round(float(log_loss(y_test, proba)), 4)
                report["brier_score"] = round(float(brier_score_loss(y_test, proba[:, 1],
                                                                      pos_label=classes[1])), 4)
                report["avg_precision"] = round(float(average_precision_score(y_test, proba[:, 1],
                                                                                pos_label=classes[1])), 4)
            else:
                report["roc_auc_ovr"] = round(float(roc_auc_score(y_test, proba, multi_class="ovr",
                                                                    average="weighted")), 4)
                report["log_loss"] = round(float(log_loss(y_test, proba)), 4)

        # Plots
        report["plots"] = {
            "confusion_matrix": self._plot_confusion_matrix(cm, model.classes_ if hasattr(model, "classes_") else None),
            "roc_curve": self._plot_roc_curve(model, X_test, y_test),
            "precision_recall_curve": self._plot_pr_curve(model, X_test, y_test),
            "calibration_plot": self._plot_calibration(model, X_test, y_test),
        }

        if X_train is not None and y_train is not None:
            report["plots"]["learning_curve"] = self._plot_learning_curve(model, X_train, y_train)

        return report

    def _plot_confusion_matrix(self, cm, labels=None) -> str:
        fig, ax = plt.subplots(figsize=(6, 5))
        _style_fig(fig, ax)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor="#2d3748",
                    cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        return _fig_to_b64(fig)

    def _plot_roc_curve(self, model, X_test, y_test) -> Optional[str]:
        if not hasattr(model, "predict_proba"):
            return None
        try:
            classes = model.classes_
            proba = model.predict_proba(X_test)
            fig, ax = plt.subplots(figsize=(6, 5))
            _style_fig(fig, ax)
            if len(classes) == 2:
                fpr, tpr, _ = roc_curve(y_test, proba[:, 1], pos_label=classes[1])
                auc = roc_auc_score(y_test, proba[:, 1])
                ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"AUC = {auc:.4f}")
            else:
                for i, cls in enumerate(classes):
                    y_bin = (y_test == cls).astype(int)
                    fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
                    auc = roc_auc_score(y_bin, proba[:, i])
                    ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=1.5,
                            label=f"Class {cls} AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
            ax.legend(framealpha=0.2, labelcolor="#e2e8f0")
            return _fig_to_b64(fig)
        except Exception as e:
            logger.warning(f"ROC curve failed: {e}")
            return None

    def _plot_pr_curve(self, model, X_test, y_test) -> Optional[str]:
        if not hasattr(model, "predict_proba"):
            return None
        try:
            classes = model.classes_
            proba = model.predict_proba(X_test)
            if len(classes) != 2:
                return None
            precision, recall, _ = precision_recall_curve(y_test, proba[:, 1], pos_label=classes[1])
            ap = average_precision_score(y_test, proba[:, 1], pos_label=classes[1])
            fig, ax = plt.subplots(figsize=(6, 5))
            _style_fig(fig, ax)
            ax.plot(recall, precision, color=PALETTE[1], lw=2, label=f"AP = {ap:.4f}")
            ax.fill_between(recall, precision, alpha=0.1, color=PALETTE[1])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
            ax.legend(framealpha=0.2, labelcolor="#e2e8f0")
            return _fig_to_b64(fig)
        except Exception as e:
            logger.warning(f"PR curve failed: {e}")
            return None

    def _plot_calibration(self, model, X_test, y_test) -> Optional[str]:
        if not hasattr(model, "predict_proba"):
            return None
        try:
            classes = model.classes_
            if len(classes) != 2:
                return None
            proba = model.predict_proba(X_test)[:, 1]
            fraction, mean_pred = calibration_curve(y_test, proba, n_bins=10, pos_label=classes[1])
            fig, ax = plt.subplots(figsize=(6, 5))
            _style_fig(fig, ax)
            ax.plot(mean_pred, fraction, "s-", color=PALETTE[0], lw=2, label="Model")
            ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1, label="Perfect calibration")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_title("Calibration Curve", fontsize=13, fontweight="bold")
            ax.legend(framealpha=0.2, labelcolor="#e2e8f0")
            return _fig_to_b64(fig)
        except Exception as e:
            logger.warning(f"Calibration plot failed: {e}")
            return None

    def _plot_learning_curve(self, model, X_train, y_train) -> str:
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy", n_jobs=1
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            _style_fig(fig, ax)
            ax.plot(train_sizes, train_scores.mean(axis=1), color=PALETTE[0], lw=2, label="Train")
            ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                             train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color=PALETTE[0])
            ax.plot(train_sizes, val_scores.mean(axis=1), color=PALETTE[2], lw=2, label="Validation")
            ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                             val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color=PALETTE[2])
            ax.set_xlabel("Training Examples")
            ax.set_ylabel("Score")
            ax.set_title("Learning Curve", fontsize=13, fontweight="bold")
            ax.legend(framealpha=0.2, labelcolor="#e2e8f0")
            return _fig_to_b64(fig)
        except Exception as e:
            logger.warning(f"Learning curve failed: {e}")
            return ""


# ─── Regression Evaluator ─────────────────────────────────────────────────────

class RegressionEvaluator:

    def full_report(self, model, X_test, y_test, X_train=None, y_train=None, feature_names=None) -> Dict:
        y_pred = model.predict(X_test)
        report = {}

        mse = mean_squared_error(y_test, y_pred)
        report["r2"] = round(float(r2_score(y_test, y_pred)), 4)
        report["adjusted_r2"] = round(float(self._adjusted_r2(r2_score(y_test, y_pred), len(y_test), X_test.shape[1])), 4)
        report["rmse"] = round(float(np.sqrt(mse)), 4)
        report["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
        report["mse"] = round(float(mse), 4)
        report["explained_variance"] = round(float(explained_variance_score(y_test, y_pred)), 4)

        try:
            report["mape"] = round(float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100), 4)
        except Exception:
            pass

        # Residuals
        residuals = np.array(y_test) - y_pred
        report["residuals_mean"] = round(float(residuals.mean()), 4)
        report["residuals_std"] = round(float(residuals.std()), 4)

        report["plots"] = {
            "actual_vs_predicted": self._plot_actual_vs_pred(y_test, y_pred),
            "residuals_plot": self._plot_residuals(y_pred, residuals),
            "residuals_distribution": self._plot_residuals_dist(residuals),
            "error_by_quantile": self._plot_error_by_quantile(y_test, y_pred),
        }

        if X_train is not None and y_train is not None:
            report["plots"]["learning_curve"] = self._plot_learning_curve(model, X_train, y_train)

        if feature_names:
            try:
                perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                report["permutation_importance"] = {
                    "features": feature_names,
                    "importances_mean": perm.importances_mean.tolist(),
                    "importances_std": perm.importances_std.tolist(),
                }
                report["plots"]["permutation_importance"] = self._plot_perm_importance(feature_names, perm)
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}")

        return report

    def _adjusted_r2(self, r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else r2

    def _plot_actual_vs_pred(self, y_test, y_pred) -> str:
        fig, ax = plt.subplots(figsize=(6, 5))
        _style_fig(fig, ax)
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, color=PALETTE[0], edgecolors="none")
        lo = min(min(y_test), min(y_pred))
        hi = max(max(y_test), max(y_pred))
        ax.plot([lo, hi], [lo, hi], "--", color="#f59e0b", lw=1.5)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted", fontsize=13, fontweight="bold")
        return _fig_to_b64(fig)

    def _plot_residuals(self, y_pred, residuals) -> str:
        fig, ax = plt.subplots(figsize=(6, 5))
        _style_fig(fig, ax)
        ax.scatter(y_pred, residuals, alpha=0.5, s=20, color=PALETTE[1], edgecolors="none")
        ax.axhline(0, color="#f59e0b", lw=1.5, linestyle="--")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals Plot", fontsize=13, fontweight="bold")
        return _fig_to_b64(fig)

    def _plot_residuals_dist(self, residuals) -> str:
        fig, ax = plt.subplots(figsize=(6, 4))
        _style_fig(fig, ax)
        ax.hist(residuals, bins=30, color=PALETTE[2], alpha=0.7, edgecolor="none")
        ax.axvline(residuals.mean(), color="#f59e0b", lw=2, label=f"Mean={residuals.mean():.2f}")
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Count")
        ax.set_title("Residuals Distribution", fontsize=13, fontweight="bold")
        ax.legend(framealpha=0.2, labelcolor="#e2e8f0")
        return _fig_to_b64(fig)

    def _plot_error_by_quantile(self, y_test, y_pred) -> str:
        fig, ax = plt.subplots(figsize=(7, 4))
        _style_fig(fig, ax)
        y_arr = np.array(y_test)
        quantiles = np.percentile(y_arr, np.arange(0, 101, 10))
        errors = []
        for i in range(len(quantiles) - 1):
            mask = (y_arr >= quantiles[i]) & (y_arr < quantiles[i+1])
            if mask.sum() > 0:
                errors.append(mean_absolute_error(y_arr[mask], y_pred[mask]))
            else:
                errors.append(0)
        ax.bar(range(len(errors)), errors, color=PALETTE[3], alpha=0.8)
        ax.set_xlabel("Decile of Actual Values")
        ax.set_ylabel("MAE")
        ax.set_title("Error by Target Value Quantile", fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(errors)))
        ax.set_xticklabels([f"D{i+1}" for i in range(len(errors))], fontsize=8)
        return _fig_to_b64(fig)

    def _plot_learning_curve(self, model, X_train, y_train) -> str:
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 8), scoring="r2", n_jobs=1
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            _style_fig(fig, ax)
            ax.plot(train_sizes, train_scores.mean(axis=1), color=PALETTE[0], lw=2, label="Train R²")
            ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                             train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color=PALETTE[0])
            ax.plot(train_sizes, val_scores.mean(axis=1), color=PALETTE[2], lw=2, label="Validation R²")
            ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                             val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color=PALETTE[2])
            ax.set_xlabel("Training Examples")
            ax.set_ylabel("R² Score")
            ax.set_title("Learning Curve", fontsize=13, fontweight="bold")
            ax.legend(framealpha=0.2, labelcolor="#e2e8f0")
            return _fig_to_b64(fig)
        except Exception as e:
            return ""

    def _plot_perm_importance(self, feature_names, perm) -> str:
        idx = np.argsort(perm.importances_mean)[::-1][:15]
        fig, ax = plt.subplots(figsize=(7, 5))
        _style_fig(fig, ax)
        cols = [feature_names[i] for i in idx]
        vals = perm.importances_mean[idx]
        errs = perm.importances_std[idx]
        ax.barh(range(len(cols)), vals[::-1], xerr=errs[::-1], color=PALETTE[4], alpha=0.8, height=0.6)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("Permutation Importance (Top 15)", fontsize=13, fontweight="bold")
        return _fig_to_b64(fig)


# ─── Feature Importance Plotter ───────────────────────────────────────────────

def plot_feature_importance(feature_names: List[str], importances: List[float], title="Feature Importance") -> str:
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_fig(fig, ax)
    top_names = [feature_names[i] for i in idx]
    top_vals = [importances[i] for i in idx]

    bars = ax.barh(range(len(top_names)), top_vals[::-1], color=PALETTE[0], alpha=0.85, height=0.6)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(title, fontsize=13, fontweight="bold")

    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#94a3b8")
    return _fig_to_b64(fig)


def plot_model_comparison(results: List[Dict], task_type: str) -> str:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return ""

    metric = "cv_mean" if task_type == "classification" else "r2"
    valid = sorted(valid, key=lambda x: x.get(metric, -999), reverse=True)[:12]

    names = [r["name"] for r in valid]
    scores = [r.get(metric, 0) for r in valid]
    times = [r.get("training_time", 0) for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _style_fig(fig, [ax1, ax2])

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax1.barh(names[::-1], scores[::-1], color=colors[::-1], alpha=0.85, height=0.6)
    ax1.set_xlabel(f"{'CV F1 Score' if task_type == 'classification' else 'R² Score'}")
    ax1.set_title("Model Performance Comparison", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, scores[::-1]):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8, color="#94a3b8")

    ax2.barh(names[::-1], times[::-1], color=PALETTE[4], alpha=0.8, height=0.6)
    ax2.set_xlabel("Training Time (seconds)")
    ax2.set_title("Training Time Comparison", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return _fig_to_b64(fig)
