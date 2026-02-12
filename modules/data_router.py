"""Data Router — Auto-ML Suite v2.0"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
import pandas as pd
import numpy as np
import io
from typing import Optional, Dict, Any

from data_preprocessing import DataAnalyzer, DataPreprocessor

router = APIRouter()

# In-memory store (replace with Redis/DB in production)
_store: Dict[str, Any] = {}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV or Excel file and return dataset profile."""
    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or Excel.")

        _store["df"] = df.to_json()

        profile = DataAnalyzer.profile(df)
        profile["shape"] = list(profile["shape"])

        return {
            "status": "ok",
            "filename": file.filename,
            "profile": _json_clean(profile),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/columns")
async def get_columns():
    """Get column names from loaded dataset."""
    if "df" not in _store:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload first.")

    df = pd.read_json(_store["df"])
    return {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


@router.post("/profile")
async def profile_dataset(target: Optional[str] = None):
    """Get full dataset profile."""
    if "df" not in _store:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    df = pd.read_json(_store["df"])
    profile = DataAnalyzer.profile(df, target)
    profile["shape"] = list(profile["shape"])

    return _json_clean(profile)


@router.post("/correlation")
async def correlation_matrix():
    """Get correlation matrix for numerical features."""
    if "df" not in _store:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    df = pd.read_json(_store["df"])
    return DataAnalyzer.correlation_matrix(df)


@router.post("/preprocess")
async def preprocess(body: Dict[str, Any] = Body(...)):
    """Preprocess the loaded dataset."""
    if "df" not in _store:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    target = body.get("target")
    config = body.get("config", {})

    df = pd.read_json(_store["df"])
    X = df.drop(columns=[target]) if target else df
    y = df[target] if target else None

    preprocessor = DataPreprocessor(
        missing_strategy=config.get("missing_strategy", "auto"),
        scaling=config.get("scaling", "standard"),
        encoding=config.get("encoding", "auto"),
        outlier_method=config.get("outlier_method", "none"),
    )

    X_proc = preprocessor.fit_transform(X, y)

    _store["X_processed"] = X_proc.to_json()
    _store["y"] = y.to_json() if y is not None else None
    _store["feature_names"] = X_proc.columns.tolist()
    _store["preprocessing_log"] = preprocessor.preprocessing_log

    return {
        "status": "ok",
        "original_shape": list(X.shape),
        "processed_shape": list(X_proc.shape),
        "features": X_proc.columns.tolist(),
        "preprocessing_log": preprocessor.preprocessing_log,
    }


def _json_clean(obj):
    """Recursively make object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj
