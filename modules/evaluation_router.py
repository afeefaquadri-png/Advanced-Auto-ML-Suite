"""Evaluation Router — Auto-ML Suite v2.0"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
import pandas as pd

from model_router import _store as model_store
from evaluation import ClassificationEvaluator, RegressionEvaluator

router = APIRouter()


@router.post("/report")
async def evaluation_report(body: Dict[str, Any] = Body(default={})):
    """Generate full evaluation report for the best model."""

    if "trainer" not in model_store:
        raise HTTPException(status_code=400, detail="No trained model. Train a model first.")

    trainer = model_store["trainer"]
    include_plots = body.get("include_plots", False)

    try:
        model = trainer.best_model
        task_type = trainer.task_type

        if task_type == "classification":
            ev = ClassificationEvaluator()
            report = ev.full_report(
                model,
                trainer.X_test,
                trainer.y_test,
                trainer.X_train,
                trainer.y_train,
            )
        else:
            ev = RegressionEvaluator()
            report = ev.full_report(
                model,
                trainer.X_test,
                trainer.y_test,
                trainer.X_train,
                trainer.y_train,
                model_store.get("feature_names"),
            )

        if not include_plots:
            report.pop("plots", None)

        return {
            "model": trainer.best_model_name,
            "task_type": task_type,
            "report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def quick_metrics():
    """Get quick metric summary from training results."""

    if "trainer" not in model_store:
        raise HTTPException(status_code=404, detail="No model trained.")

    trainer = model_store["trainer"]

    if trainer.task_type == "classification":
        from sklearn.metrics import accuracy_score, f1_score

        y_pred = trainer.best_model.predict(trainer.X_test)
        return {
            "model": trainer.best_model_name,
            "accuracy": round(float(accuracy_score(trainer.y_test, y_pred)), 4),
            "f1_weighted": round(
                float(f1_score(trainer.y_test, y_pred, average="weighted", zero_division=0)),
                4,
            ),
        }

    else:
        from sklearn.metrics import r2_score, mean_squared_error
        import numpy as np

        y_pred = trainer.best_model.predict(trainer.X_test)
        return {
            "model": trainer.best_model_name,
            "r2": round(float(r2_score(trainer.y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(trainer.y_test, y_pred))), 4),
        }
