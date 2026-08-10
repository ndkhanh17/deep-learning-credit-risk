"""
src/feature_engineering.py
===========================
Feature engineering pipeline cho du lieu tin dung:
- Ratio features
- EXT_SOURCE interactions (~25 features)
- Target Encoding voi Bayesian smoothing (~30 features)
- Missing indicators
- QuantileTransformer
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from src.config import TARGET_ENCODE_COLS, TE_SMOOTH, MISSING_THRESHOLD, SEED


# ============================================================
# BUOC 1: LABEL ENCODE CATEGORICAL
# ============================================================
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode tat ca bien phan loai (object dtype)."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[FE] Encoded {len(cat_cols)} categorical columns")
    return df


# ============================================================
# BUOC 2: LOAI BO BIEN NHIEU (MISSING > THRESHOLD)
# ============================================================
def drop_high_missing(df: pd.DataFrame,
                       threshold: float = MISSING_THRESHOLD,
                       exclude: list = None) -> pd.DataFrame:
    """Loai bo cac cot co ty le missing > threshold."""
    exclude = exclude or ["TARGET", "SK_ID_CURR"]
    miss_rate = df.isnull().mean()
    drop_cols = [c for c in miss_rate[miss_rate > threshold].index
                 if c not in exclude]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"[FE] Dropped {len(drop_cols)} columns (missing > {threshold:.0%})")
    return df


# ============================================================
# BUOC 3: RATIO FEATURES CO BAN
# ============================================================
def build_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tao cac ratio features co ban tu bien tai chinh."""
    df = df.copy()
    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-5)
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-5)
    df["AGE_YEARS"]            = -df["DAYS_BIRTH"] / 365
    df["EMPLOYED_YEARS"]       = (-df["DAYS_EMPLOYED"].clip(upper=0)) / 365
    df["EMPLOYED_AGE_RATIO"]   = df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + 1e-5)
    df["INCOME_PER_PERSON"]    = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(lower=1)
    df["GOODS_CREDIT_RATIO"]   = df.get("AMT_GOODS_PRICE", 0) / (df["AMT_CREDIT"] + 1e-5)
    df["CREDIT_TERM_MONTHS"]   = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-5)
    print("[FE] Built 9 ratio features")
    return df


# ============================================================
# BUOC 4: EXT_SOURCE INTERACTIONS (~25 features)
# ============================================================
def build_ext_source_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tao cac tuong tac bac 2 va bac 3 tu EXT_SOURCE_1/2/3.
    Day la nhom features quan trong nhat theo bao cao.
    """
    df = df.copy()
    exts = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            if c in df.columns]
    n_before = df.shape[1]

    # Bac 2: pairwise interactions
    for i, c1 in enumerate(exts):
        for c2 in exts[i+1:]:
            df[f"{c1}_x_{c2}"]    = df[c1] * df[c2]
            df[f"{c1}_div_{c2}"]  = df[c1] / (df[c2] + 1e-5)
            df[f"{c1}_plus_{c2}"] = df[c1] + df[c2]
            df[f"{c1}_minus_{c2}"] = df[c1] - df[c2]

        # Tu dong
        df[f"{c1}_sq"]   = df[c1] ** 2
        df[f"{c1}_cube"] = df[c1] ** 3

    # Thong ke tong hop EXT_SOURCE
    if len(exts) >= 2:
        df["EXT_MEAN"] = df[exts].mean(axis=1)
        df["EXT_MIN"]  = df[exts].min(axis=1)
        df["EXT_MAX"]  = df[exts].max(axis=1)
        df["EXT_STD"]  = df[exts].std(axis=1)
        df["EXT_RANGE"] = df["EXT_MAX"] - df["EXT_MIN"]

    # Cross-domain interactions
    if "EXT_SOURCE_2" in df.columns and "CREDIT_INCOME_RATIO" in df.columns:
        df["EXT2_x_CREDIT_INCOME"] = df["EXT_SOURCE_2"] * df["CREDIT_INCOME_RATIO"]
    if "EXT_SOURCE_3" in df.columns and "AGE_YEARS" in df.columns:
        df["EXT3_x_AGE"]           = df["EXT_SOURCE_3"] * df["AGE_YEARS"]
    if "EXT_SOURCE_1" in df.columns and "DAYS_BIRTH" in df.columns:
        df["EXT1_x_DAYS_BIRTH"]    = df["EXT_SOURCE_1"] * df["DAYS_BIRTH"]

    n_added = df.shape[1] - n_before
    print(f"[FE] Built {n_added} EXT_SOURCE interaction features")
    return df


# ============================================================
# BUOC 5: TARGET ENCODING (Bayesian Smoothing)
# ============================================================
def apply_target_encoding(df: pd.DataFrame, y: pd.Series,
                            cols: list = TARGET_ENCODE_COLS,
                            m: int = TE_SMOOTH) -> pd.DataFrame:
    """
    Target Encoding voi Bayesian smoothing:
      TE(c) = (n_c * mean_c + m * global_mean) / (n_c + m)
    m = smoothing factor (default=20)
    """
    df = df.copy()
    global_mean = y.mean()
    n_added = 0

    for col in cols:
        if col not in df.columns:
            continue
        cat_stats = y.groupby(df[col]).agg(["mean", "count"])
        cat_stats["te"] = ((cat_stats["count"] * cat_stats["mean"] +
                             m * global_mean) / (cat_stats["count"] + m))
        te_map = cat_stats["te"].to_dict()
        df[f"TE_{col}"] = df[col].map(te_map).fillna(global_mean)
        n_added += 1

    # Cross TE x EXT_SOURCE
    if "TE_ORGANIZATION_TYPE" in df.columns and "EXT_SOURCE_2" in df.columns:
        df["TE_ORG_x_EXT2"] = df["TE_ORGANIZATION_TYPE"] * df["EXT_SOURCE_2"]
    if "TE_OCCUPATION_TYPE" in df.columns and "EXT_SOURCE_3" in df.columns:
        df["TE_OCC_x_EXT3"] = df["TE_OCCUPATION_TYPE"] * df["EXT_SOURCE_3"]

    print(f"[FE] Applied Target Encoding for {n_added} columns + 2 cross features")
    return df


# ============================================================
# BUOC 6: MISSING INDICATORS
# ============================================================
def add_missing_indicators(df: pd.DataFrame,
                             cols: list = None) -> pd.DataFrame:
    """Them cot 0/1 chi dinh xem bien co missing hay khong."""
    df = df.copy()
    if cols is None:
        cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
                if c in df.columns]
    for col in cols:
        df[f"{col}_NA"] = df[col].isna().astype(int)
    print(f"[FE] Added {len(cols)} missing indicator columns")
    return df


# ============================================================
# BUOC 7: IMPUTE + QUANTILE TRANSFORM
# ============================================================
def prepare_features(df: pd.DataFrame,
                      exclude_cols: list = None,
                      seed: int = SEED):
    """
    Tach features va label, median impute, QuantileTransformer.
    Returns: X_scaled (np.array), y (np.array), feature_names (list),
             imputer, scaler
    """
    exclude_cols = exclude_cols or ["SK_ID_CURR", "TARGET"]
    y = df["TARGET"].values

    feat_cols = [c for c in df.columns if c not in exclude_cols]
    # Loai bien variance = 0
    variances = df[feat_cols].var()
    zero_var  = variances[variances < 1e-10].index.tolist()
    feat_cols = [c for c in feat_cols if c not in zero_var]

    X_raw = df[feat_cols].values

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    scaler  = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(X_imp)),
        random_state=seed
    )
    X_scaled = scaler.fit_transform(X_imp)

    print(f"[FE] Final features: {X_scaled.shape[1]} | "
          f"Removed {len(zero_var)} zero-variance cols")
    print(f"[FE] X mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f}")
    return X_scaled, y, feat_cols, imputer, scaler


# ============================================================
# PIPELINE DAY DU
# ============================================================
def build_full_pipeline(df: pd.DataFrame) -> tuple:
    """
    Chay toan bo pipeline Feature Engineering.
    Input:  df - sau khi merge tat ca bang phu
    Output: X_scaled, y, feat_cols, imputer, scaler
    """
    print("\n" + "="*55)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*55)

    y_series = df["TARGET"]
    n0 = df.shape[1]

    df = encode_categoricals(df)
    df = drop_high_missing(df)
    df = build_ratio_features(df)
    df = build_ext_source_interactions(df)
    df = apply_target_encoding(df, y_series)
    df = add_missing_indicators(df)

    print(f"\n[FE] Features: {n0} -> {df.shape[1]} (+{df.shape[1]-n0} moi)")
    X, y, feat_cols, imputer, scaler = prepare_features(df)
    return X, y, feat_cols, imputer, scaler, df
