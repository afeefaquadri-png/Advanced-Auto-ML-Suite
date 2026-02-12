"""Model Router — Auto-ML Suite v2.0"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.model_training import AutoMLTrainer

router = APIRouter()
_store: Dict[str, Any] = {}


@router.post("/train")
async def train_model(body: Dict[str, Any] = Body(...)):
    """
    Train all registered algorithms and return ranked results.
    
    Body:
    - target (str): target column name
    - task_type (str): 'auto' | 'classification' | 'regression'
    - cv_folds (int): number of CV folds (default 5)
    - test_size (float): test split ratio (default 0.2)
    - tune_hyperparams (bool): run hyperparameter search (default true)
    - tuning_method (str): 'random' | 'grid'
    - n_iter (int): random search iterations
    - selected_algorithms (list): optional list of algorithm names
    """
    try:
        from backend.routers.data_router import _store as data_store
    except ImportError:
        raise HTTPException(status_code=400, detail="Data not loaded. Upload and preprocess data first.")

    if "X_processed" not in data_store:
        raise HTTPException(status_code=400, detail="Preprocessed data not found. Run /api/data/preprocess first.")

    X = pd.read_json(data_store["X_processed"])
    y_json = data_store.get("y")
    if y_json is None:
        raise HTTPException(status_code=400, detail="Target column not found in preprocessed data.")
    y = pd.read_json(y_json, typ="series")

    trainer = AutoMLTrainer(
        task_type=body.get("task_type", "auto"),
        test_size=body.get("test_size", 0.2),
        cv_folds=body.get("cv_folds", 5),
        tune_hyperparams=body.get("tune_hyperparams", True),
        tuning_method=body.get("tuning_method", "random"),
        n_iter=body.get("n_iter", 10),
        selected_algorithms=body.get("selected_algorithms", None),
    )

    results = trainer.train(X, y)
    _store["trainer"] = trainer
    _store["feature_names"] = data_store.get("feature_names", [])

    # Clean results for JSON
    clean_results = [
        {k: v for k, v in r.items() if k not in ["confusion_matrix", "classification_report"]}
        for r in results["results"]
    ]
    return {**results, "results": clean_results}


@router.post("/predict")
async def predict(body: Dict[str, Any] = Body(...)):
    """
    Make predictions using the best trained model.
    
    Body:
    - features (dict | list[dict]): input feature values
    - return_proba (bool): return probabilities for classification
    """
    if "trainer" not in _store:
        raise HTTPException(status_code=400, detail="No trained model found. Train a model first.")

    trainer = _store["trainer"]
    features = body.get("features", {})
    return_proba = body.get("return_proba", False)

    if isinstance(features, dict):
        X = pd.DataFrame([features])
    elif isinstance(features, list):
        X = pd.DataFrame(features)
    else:
        raise HTTPException(status_code=400, detail="'features' must be a dict or list of dicts.")

    try:
        preds = trainer.predict(X)
        result: Dict[str, Any] = {
            "model": trainer.best_model_name,
            "task_type": trainer.task_type,
            "predictions": preds.tolist(),
            "n_predictions": len(preds),
        }
        if return_proba and trainer.task_type == "classification":
            proba = trainer.predict_proba(X)
            result["probabilities"] = proba.tolist()
            if hasattr(trainer.best_model, "classes_"):
                result["classes"] = trainer.best_model.classes_.tolist()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/algorithms")
async def list_algorithms(task_type: str = "classification"):
    """List all available algorithms for a given task type."""
    from modules.model_training import get_classifier_registry, get_regressor_registry
    reg = get_classifier_registry() if task_type == "classification" else get_regressor_registry()
    return {
        "task_type": task_type,
        "algorithms": [
            {"name": name, "category": info["category"]}
            for name, info in reg.items()
        ]
    }


@router.get("/model/info")
async def model_info():
    """Get info about the currently trained best model."""
    if "trainer" not in _store:
        raise HTTPException(status_code=404, detail="No model trained yet.")
    trainer = _store["trainer"]
    fi = trainer.get_feature_importance()
    return {
        "name": trainer.best_model_name,
        "task_type": trainer.task_type,
        "feature_names": _store.get("feature_names", []),
        "feature_importance": fi,
    }


@router.post("/model/save")
async def save_model(path: str = "models/best_model.pkl"):
    """Save the best model to disk."""
    if "trainer" not in _store:
        raise HTTPException(status_code=404, detail="No model trained yet.")
    saved_path = _store["trainer"].save_model(path)
    return {"status": "saved", "path": saved_path}
