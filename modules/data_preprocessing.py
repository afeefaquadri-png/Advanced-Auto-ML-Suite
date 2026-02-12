"""
Data Preprocessing & Feature Engineering — Auto-ML Suite v2.0
Handles missing values, encoding, scaling, outliers, feature selection, and generation.
"""

# REQUIRED for IterativeImputer (must be first sklearn-related import)
from sklearn.experimental import enable_iterative_imputer  # noqa

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    PowerTransformer,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)

from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer,
)

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    VarianceThreshold,
    RFECV,
    SelectFromModel,
)

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging

logger = logging.getLogger(__name__)



class DataAnalyzer:
    """Comprehensive dataset analysis and profiling."""

    @staticmethod
    def profile(df: pd.DataFrame, target: Optional[str] = None) -> Dict:
        profile = {
            "shape": df.shape,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "columns": {},
            "missing": {},
            "duplicates": int(df.duplicated().sum()),
            "target_info": None,
        }

        for col in df.columns:
            dtype = str(df[col].dtype)
            col_info = {
                "dtype": dtype,
                "n_missing": int(df[col].isna().sum()),
                "pct_missing": round(df[col].isna().mean() * 100, 2),
                "n_unique": int(df[col].nunique()),
            }

            if np.issubdtype(df[col].dtype, np.number):
                stats = df[col].describe()
                col_info.update({
                    "kind": "numerical",
                    "mean": round(float(stats["mean"]), 4),
                    "std": round(float(stats["std"]), 4),
                    "min": round(float(stats["min"]), 4),
                    "q25": round(float(stats["25%"]), 4),
                    "median": round(float(stats["50%"]), 4),
                    "q75": round(float(stats["75%"]), 4),
                    "max": round(float(stats["max"]), 4),
                    "skewness": round(float(df[col].skew()), 4),
                    "kurtosis": round(float(df[col].kurtosis()), 4),
                    "n_zeros": int((df[col] == 0).sum()),
                    "n_negative": int((df[col] < 0).sum()),
                    "n_outliers_iqr": DataAnalyzer._count_iqr_outliers(df[col]),
                    "sample": df[col].dropna().sample(min(5, df[col].count()), random_state=42).tolist(),
                })
            else:
                top = df[col].value_counts().head(5)
                col_info.update({
                    "kind": "categorical",
                    "top_values": top.to_dict(),
                    "sample": df[col].dropna().sample(min(5, df[col].count()), random_state=42).tolist(),
                })

            profile["columns"][col] = col_info

        profile["missing"] = {
            col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0
        }

        if target and target in df.columns:
            y = df[target]
            profile["target_info"] = {
                "column": target,
                "dtype": str(y.dtype),
                "n_unique": int(y.nunique()),
                "value_counts": y.value_counts().head(20).to_dict(),
                "suggested_task": "classification" if (y.dtype == "object" or y.nunique() <= 20) else "regression",
            }

        return profile

    @staticmethod
    def _count_iqr_outliers(series: pd.Series) -> int:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

    @staticmethod
    def correlation_matrix(df: pd.DataFrame) -> Dict:
        num_df = df.select_dtypes(include=np.number)
        if num_df.empty:
            return {}
        corr = num_df.corr()
        return {
            "columns": corr.columns.tolist(),
            "matrix": corr.values.tolist(),
        }

    @staticmethod
    def detect_high_correlation(df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = [
            (col, row, round(float(upper.loc[row, col]), 4))
            for col in upper.columns
            for row in upper.index
            if pd.notna(upper.loc[row, col]) and upper.loc[row, col] >= threshold
        ]
        return pairs


class DataPreprocessor:
    """Full ML preprocessing pipeline with configurable strategies."""

    def __init__(self,
                 missing_strategy: str = "auto",
                 scaling: str = "standard",
                 encoding: str = "auto",
                 outlier_method: str = "none",
                 outlier_threshold: float = 3.0,
                 drop_high_corr: bool = False,
                 corr_threshold: float = 0.95,
                 drop_low_variance: bool = False,
                 variance_threshold: float = 0.0):

        self.missing_strategy = missing_strategy
        self.scaling = scaling
        self.encoding = encoding
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.drop_high_corr = drop_high_corr
        self.corr_threshold = corr_threshold
        self.drop_low_variance = drop_low_variance
        self.variance_threshold = variance_threshold

        self._scalers: Dict[str, Any] = {}
        self._imputers: Dict[str, Any] = {}
        self._encoders: Dict[str, Any] = {}
        self._dropped_cols: List[str] = []
        self._label_cols: List[str] = []
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._is_fitted = False
        self.preprocessing_log: List[str] = []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        df = X.copy()
        self._num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self._cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        self.preprocessing_log = []

        # ── Drop high-correlation ──────────────────────────────────────────────
        if self.drop_high_corr:
            pairs = DataAnalyzer.detect_high_correlation(df, self.corr_threshold)
            to_drop = list({p[1] for p in pairs})
            df.drop(columns=to_drop, inplace=True, errors="ignore")
            self._dropped_cols.extend(to_drop)
            self._num_cols = [c for c in self._num_cols if c not in to_drop]
            self.preprocessing_log.append(f"Dropped {len(to_drop)} highly correlated columns: {to_drop}")

        # ── Outlier handling ──────────────────────────────────────────────────
        if self.outlier_method != "none" and self._num_cols:
            df = self._handle_outliers(df)

        # ── Missing values — numerical ─────────────────────────────────────────
        if self._num_cols:
            df = self._impute_numerical(df)

        # ── Missing values — categorical ──────────────────────────────────────
        if self._cat_cols:
            df = self._impute_categorical(df)

        # ── Encoding ─────────────────────────────────────────────────────────
        if self._cat_cols:
            df = self._encode_categorical(df, y)

        # ── Scaling ─────────────────────────────────────────────────────────
        updated_num = df.select_dtypes(include=np.number).columns.tolist()
        if updated_num and self.scaling != "none":
            df = self._scale_numerical(df, updated_num)

        # ── Low variance ─────────────────────────────────────────────────────
        if self.drop_low_variance:
            selector = VarianceThreshold(threshold=self.variance_threshold)
            arr = selector.fit_transform(df)
            kept = df.columns[selector.get_support()].tolist()
            removed = [c for c in df.columns if c not in kept]
            df = pd.DataFrame(arr, columns=kept, index=df.index)
            self.preprocessing_log.append(f"Dropped {len(removed)} low-variance features: {removed}")

        self._is_fitted = True
        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        df = X.copy()
        df.drop(columns=self._dropped_cols, inplace=True, errors="ignore")

        for col, imputer in self._imputers.items():
            if col in df.columns:
                df[col] = imputer.transform(df[[col]])

        for col, enc in self._encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col])

        for col, scaler in self._scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_method == "zscore":
            z = np.abs((df[self._num_cols] - df[self._num_cols].mean()) / df[self._num_cols].std())
            df[self._num_cols] = df[self._num_cols].where(z < self.outlier_threshold, np.nan)
            self.preprocessing_log.append(f"Z-score outliers capped to NaN (threshold={self.outlier_threshold})")
        elif self.outlier_method == "iqr":
            for col in self._num_cols:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            self.preprocessing_log.append("IQR clipping applied to numerical columns")
        elif self.outlier_method == "winsorize":
            for col in self._num_cols:
                lo, hi = df[col].quantile(0.05), df[col].quantile(0.95)
                df[col] = df[col].clip(lo, hi)
            self.preprocessing_log.append("Winsorization (5th–95th pct) applied")
        return df

    def _impute_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [c for c in self._num_cols if df[c].isna().any()]
        if not missing_cols:
            return df

        if self.missing_strategy == "auto":
            strategy = "knn" if len(missing_cols) <= 10 else "median"
        else:
            strategy = self.missing_strategy

        if strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
            df[self._num_cols] = imputer.fit_transform(df[self._num_cols])
            self.preprocessing_log.append(f"KNN imputation on {len(missing_cols)} numerical columns")
        elif strategy == "iterative":
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df[self._num_cols] = imputer.fit_transform(df[self._num_cols])
            self.preprocessing_log.append("Iterative imputation on numerical columns")
        else:
            imputer = SimpleImputer(strategy=strategy if strategy in ["mean", "median", "most_frequent"] else "median")
            df[self._num_cols] = imputer.fit_transform(df[self._num_cols])
            self.preprocessing_log.append(f"SimpleImputer ({strategy}) on {len(missing_cols)} numerical columns")

        return df

    def _impute_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._cat_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        self.preprocessing_log.append(f"Mode imputation on categorical columns: {self._cat_cols}")
        return df

    def _encode_categorical(self, df: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        for col in self._cat_cols:
            if col not in df.columns:
                continue
            n_unique = df[col].nunique()
            strategy = self.encoding

            if strategy == "auto":
                if n_unique == 2:
                    strategy = "label"
                elif n_unique <= 10:
                    strategy = "onehot"
                else:
                    strategy = "label"

            if strategy == "onehot":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.preprocessing_log.append(f"One-hot encoded '{col}' → {dummies.shape[1]} columns")
            elif strategy == "label":
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].astype(str))
                self._encoders[col] = enc
                self.preprocessing_log.append(f"Label encoded '{col}'")
            elif strategy == "ordinal":
                enc = OrdinalEncoder()
                df[col] = enc.fit_transform(df[[col]])
                self._encoders[col] = enc
                self.preprocessing_log.append(f"Ordinal encoded '{col}'")

        return df

    def _scale_numerical(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler(),
            "power": PowerTransformer(),
        }
        scaler = scaler_map.get(self.scaling, StandardScaler())
        df[num_cols] = scaler.fit_transform(df[num_cols])
        self.preprocessing_log.append(f"Applied {self.scaling} scaling to {len(num_cols)} numerical columns")
        return df


class FeatureEngineer:
    """Feature generation, selection, and dimensionality reduction."""

    @staticmethod
    def select_features(X: pd.DataFrame, y: pd.Series, task: str = "classification",
                        method: str = "mutual_info", k: int = 10) -> Tuple[pd.DataFrame, List[str], Dict]:
        if k >= X.shape[1]:
            return X, X.columns.tolist(), {"selected_k": X.shape[1]}

        if method == "mutual_info":
            score_fn = mutual_info_classif if task == "classification" else mutual_info_regression
        elif method == "f_test":
            score_fn = f_classif if task == "classification" else f_regression
        elif method == "random_forest":
            rf = (RandomForestClassifier(n_estimators=50, random_state=42)
                  if task == "classification"
                  else RandomForestRegressor(n_estimators=50, random_state=42))
            rf.fit(X, y)
            selector = SelectFromModel(rf, max_features=k, prefit=True)
            mask = selector.get_support()
            selected = X.columns[mask].tolist()
            scores = dict(zip(X.columns.tolist(), rf.feature_importances_.tolist()))
            return X[selected], selected, {"method": "random_forest", "scores": scores}
        else:
            score_fn = mutual_info_classif if task == "classification" else mutual_info_regression

        selector = SelectKBest(score_fn, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        scores = dict(zip(X.columns.tolist(), selector.scores_.tolist()))
        return X[selected], selected, {"method": method, "scores": scores}

    @staticmethod
    def apply_pca(X: pd.DataFrame, n_components: int = 5) -> Tuple[pd.DataFrame, Dict]:
        n = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n, random_state=42)
        transformed = pca.fit_transform(X)
        cols = [f"PC{i+1}" for i in range(n)]
        df_pca = pd.DataFrame(transformed, columns=cols, index=X.index)
        return df_pca, {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "n_components": n,
        }

    @staticmethod
    def generate_polynomial_features(X: pd.DataFrame, degree: int = 2,
                                      cols: Optional[List[str]] = None) -> pd.DataFrame:
        from sklearn.preprocessing import PolynomialFeatures
        target_cols = cols or X.select_dtypes(include=np.number).columns.tolist()[:5]
        if not target_cols:
            return X
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        arr = poly.fit_transform(X[target_cols])
        names = poly.get_feature_names_out(target_cols)
        new_df = pd.DataFrame(arr, columns=names, index=X.index)
        other_cols = [c for c in X.columns if c not in target_cols]
        return pd.concat([X[other_cols], new_df], axis=1)

    @staticmethod
    def generate_interaction_features(X: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        num_cols = cols or X.select_dtypes(include=np.number).columns.tolist()[:6]
        new_features = {}
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i+1:]:
                new_features[f"{c1}_x_{c2}"] = X[c1] * X[c2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(X[c2] != 0, X[c1] / X[c2], 0)
                    new_features[f"{c1}_div_{c2}"] = ratio
        return pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
