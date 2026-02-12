from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

import data_router
import model_router
import evaluation_router
import feature_router
import experiment_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Auto-ML Suite API",
    description="Comprehensive AutoML platform with advanced evaluation, feature engineering, and experiment tracking.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router.router, prefix="/api/data", tags=["Data"])
app.include_router(model_router.router, prefix="/api/model", tags=["Model Training"])
app.include_router(evaluation_router.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(feature_router.router, prefix="/api/features", tags=["Feature Engineering"])
app.include_router(experiment_router.router, prefix="/api/experiments", tags=["Experiments"])

@app.get("/")
async def root():
    return {"status": "ok", "message": "Auto-ML Suite v2.0 is running.", "docs": "/api/docs"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


