"""Experiment Tracking Router — Auto-ML Suite v2.0"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List
import json
import time
import uuid
from datetime import datetime

router = APIRouter()

# In-memory experiment store (replace with MLflow/W&B/DB in production)
_experiments: List[Dict] = []


@router.post("/log")
async def log_experiment(body: Dict[str, Any] = Body(...)):
    """Log a new experiment run."""
    exp = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "name": body.get("name", f"run_{len(_experiments)+1}"),
        "task_type": body.get("task_type"),
        "best_model": body.get("best_model"),
        "metrics": body.get("metrics", {}),
        "config": body.get("config", {}),
        "dataset_info": body.get("dataset_info", {}),
        "tags": body.get("tags", []),
    }
    _experiments.append(exp)
    return {"status": "logged", "experiment_id": exp["id"]}


@router.get("/list")
async def list_experiments():
    """List all logged experiments."""
    return {"experiments": _experiments, "total": len(_experiments)}


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get a specific experiment by ID."""
    exp = next((e for e in _experiments if e["id"] == experiment_id), None)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")
    return exp


@router.delete("/clear")
async def clear_experiments():
    """Clear all experiment logs."""
    _experiments.clear()
    return {"status": "cleared"}


@router.get("/compare/models")
async def compare_models():
    """Compare metrics across all logged experiments."""
    if not _experiments:
        return {"message": "No experiments logged yet."}

    comparison = []
    for exp in _experiments:
        row = {"id": exp["id"], "name": exp["name"], "model": exp.get("best_model"), "task": exp.get("task_type")}
        row.update(exp.get("metrics", {}))
        comparison.append(row)

    return {"comparison": comparison}
