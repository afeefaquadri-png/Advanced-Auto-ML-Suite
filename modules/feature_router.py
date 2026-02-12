"""Feature Engineering Router — Auto-ML Suite v2.0"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter()


@router.post("/select")
async def select_features(body: Dict[str, Any] = Body(...)):
    """
    Select top-K features using statistical methods.
    
    Body:
    - method: 'mutual_info' | 'f_test' | 'random_forest'
    - k: number of top features to select
    - task_type: 'classification' | 'regression'
    """
    try:
        from backend.routers.data_router import _store as data_store
        from modules.data_preprocessing import FeatureEngineer
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "X_processed" not in data_store:
        raise HTTPException(status_code=400, detail="Preprocessed data not found.")

    X = pd.read_json(data_store["X_processed"])
    y = pd.read_json(data_store.get("y", "{}"), typ="series")

    method = body.get("method", "mutual_info")
    k = body.get("k", 10)
    task_type = body.get("task_type", "classification")

    fe = FeatureEngineer()
    X_sel, selected, info = fe.select_features(X, y, task_type, method, k)

    data_store["X_processed"] = X_sel.to_json()
    data_store["feature_names"] = selected

    return {
        "selected_features": selected,
        "n_selected": len(selected),
        "method": method,
        "info": {k: v for k, v in info.items() if k != "scores"},
    }


@router.post("/pca")
async def apply_pca(body: Dict[str, Any] = Body(...)):
    """Apply PCA dimensionality reduction."""
    try:
        from backend.routers.data_router import _store as data_store
        from modules.data_preprocessing import FeatureEngineer
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "X_processed" not in data_store:
        raise HTTPException(status_code=400, detail="Preprocessed data not found.")

    X = pd.read_json(data_store["X_processed"])
    n_components = body.get("n_components", 5)

    fe = FeatureEngineer()
    X_pca, pca_info = fe.apply_pca(X, n_components)

    data_store["X_processed"] = X_pca.to_json()
    data_store["feature_names"] = X_pca.columns.tolist()

    return {"status": "ok", "shape": list(X_pca.shape), **pca_info}
