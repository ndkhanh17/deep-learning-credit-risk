"""
src/data_loader.py
==================
Load va aggregate 7 bang du lieu Home Credit Default Risk
Bang chinh: application_train, application_test
Bang phu:   bureau, bureau_balance, previous_application,
            installments_payments, POS_CASH_balance, credit_card_balance
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_DIR, FULL_DATA, SAMPLE_SIZE, MAX_ROWS_AUX, SEED


# ============================================================
# LOAD BANG CHINH
# ============================================================
def load_main_tables(full_data: bool = FULL_DATA,
                     sample_size: int = SAMPLE_SIZE,
                     seed: int = SEED):
    """Load application_train & application_test."""
    app_train = pd.read_csv(DATA_DIR / "application_train.csv")
    app_test  = pd.read_csv(DATA_DIR / "application_test.csv")

    if not full_data:
        app_train = (app_train
                     .sample(n=min(sample_size, len(app_train)), random_state=seed)
                     .reset_index(drop=True))

    print(f"[DataLoader] application_train : {app_train.shape}")
    print(f"[DataLoader] application_test  : {app_test.shape}")

    tc = app_train["TARGET"].value_counts()
    print(f"[DataLoader] TARGET=0 (Normal) : {tc[0]:,} ({tc[0]/len(app_train)*100:.1f}%)")
    print(f"[DataLoader] TARGET=1 (Default): {tc[1]:,} ({tc[1]/len(app_train)*100:.1f}%)")
    print(f"[DataLoader] Imbalance ratio   : {tc[0]/tc[1]:.1f}:1")
    return app_train, app_test


# ============================================================
# LOAD BANG PHU (CO LOC THEO valid_ids)
# ============================================================
def _load_filtered(path: Path, key_col: str, valid_ids: set,
                   max_rows: int = None, seed: int = SEED) -> pd.DataFrame:
    """Load CSV va loc chi lay rows co key trong valid_ids."""
    df = pd.read_csv(path)
    df = df[df[key_col].isin(valid_ids)].copy()
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return df


def load_auxiliary_tables(valid_ids: set,
                          full_data: bool = FULL_DATA,
                          seed: int = SEED):
    """Load 6 bang phu, chi lay du lieu cua khach hang trong valid_ids."""
    mx = None if full_data else MAX_ROWS_AUX

    bureau       = _load_filtered(DATA_DIR / "bureau.csv",
                                   "SK_ID_CURR", valid_ids, mx, seed)
    bureau_ids   = set(bureau["SK_ID_BUREAU"])
    bureau_bal   = _load_filtered(DATA_DIR / "bureau_balance.csv",
                                   "SK_ID_BUREAU", bureau_ids, mx, seed)
    prev_app     = _load_filtered(DATA_DIR / "previous_application.csv",
                                   "SK_ID_CURR", valid_ids, mx, seed)
    installments = _load_filtered(DATA_DIR / "installments_payments.csv",
                                   "SK_ID_CURR", valid_ids, mx, seed)
    pos_cash     = _load_filtered(DATA_DIR / "POS_CASH_balance.csv",
                                   "SK_ID_CURR", valid_ids, mx, seed)
    credit_card  = _load_filtered(DATA_DIR / "credit_card_balance.csv",
                                   "SK_ID_CURR", valid_ids, mx, seed)

    for name, df in [("bureau", bureau), ("bureau_balance", bureau_bal),
                     ("previous_application", prev_app),
                     ("installments_payments", installments),
                     ("POS_CASH_balance", pos_cash),
                     ("credit_card_balance", credit_card)]:
        print(f"[DataLoader] {name:25s}: {df.shape}")

    return bureau, bureau_bal, prev_app, installments, pos_cash, credit_card


# ============================================================
# AGGREGATION TUNG BANG PHU
# ============================================================
def agg_bureau(bureau: pd.DataFrame, bureau_bal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau + bureau_balance -> 1 dong / khach hang."""
    # Bureau balance stats theo SK_ID_BUREAU
    bb = bureau_bal.groupby("SK_ID_BUREAU").agg(
        BB_DPD_MEAN   = ("MONTHS_BALANCE", "mean"),
        BB_CLOSED_RATIO = ("STATUS", lambda x: (x == "C").mean()),
    ).reset_index()

    bur = bureau.merge(bb, on="SK_ID_BUREAU", how="left")

    agg = bur.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT    = ("SK_ID_BUREAU", "count"),
        BUREAU_CREDIT_SUM    = ("AMT_CREDIT_SUM", "sum"),
        BUREAU_CREDIT_MEAN   = ("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_MAX    = ("AMT_CREDIT_SUM", "max"),
        BUREAU_DEBT_SUM      = ("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_OVERDUE_MEAN  = ("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BUREAU_DPD_MEAN      = ("CREDIT_DAY_OVERDUE", "mean"),
        BUREAU_DPD_MAX       = ("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_ACTIVE_COUNT  = ("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_CLOSED_COUNT  = ("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        BUREAU_DAYS_CREDIT_MEAN = ("DAYS_CREDIT", "mean"),
        BUREAU_BB_DPD_MEAN   = ("BB_DPD_MEAN", "mean"),
    ).reset_index()

    agg["BUREAU_DEBT_RATIO"]   = agg["BUREAU_DEBT_SUM"] / (agg["BUREAU_CREDIT_SUM"] + 1e-5)
    agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_ACTIVE_COUNT"] / (agg["BUREAU_LOAN_COUNT"] + 1e-5)
    return agg


def agg_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application -> 1 dong / khach hang."""
    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT       = ("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT  = ("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        PREV_REFUSED_COUNT   = ("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        PREV_AMT_CREDIT_MEAN = ("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_MAX  = ("AMT_CREDIT", "max"),
        PREV_AMT_DOWN_MEAN   = ("AMT_DOWN_PAYMENT", "mean"),
        PREV_DAYS_LAST_DUE_MEAN = ("DAYS_LAST_DUE", "mean"),
    ).reset_index()

    agg["PREV_APPROVED_RATIO"] = (agg["PREV_APPROVED_COUNT"] /
                                    agg["PREV_APP_COUNT"].clip(lower=1))
    agg["PREV_REFUSED_RATIO"]  = (agg["PREV_REFUSED_COUNT"] /
                                    agg["PREV_APP_COUNT"].clip(lower=1))
    return agg


def agg_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments -> 1 dong / khach hang."""
    df = inst.copy()
    df["DAYS_LATE"]  = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["PAY_DIFF"]   = df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]
    df["IS_LATE"]    = (df["DAYS_LATE"] > 0).astype(int)
    df["IS_UNDERPAY"] = (df["PAY_DIFF"] < 0).astype(int)

    agg = df.groupby("SK_ID_CURR").agg(
        INST_COUNT          = ("NUM_INSTALMENT_NUMBER", "count"),
        INST_LATE_MEAN      = ("DAYS_LATE", "mean"),
        INST_LATE_MAX       = ("DAYS_LATE", "max"),
        INST_LATE_RATIO     = ("IS_LATE", "mean"),
        INST_PAY_DIFF_MEAN  = ("PAY_DIFF", "mean"),
        INST_UNDERPAY_RATIO = ("IS_UNDERPAY", "mean"),
    ).reset_index()

    # Installments trong 12 thang gan nhat
    recent = df[df["DAYS_INSTALMENT"] >= -365]
    if len(recent) > 0:
        recent_agg = recent.groupby("SK_ID_CURR").agg(
            INST_RECENT_LATE_RATIO = ("IS_LATE", "mean"),
            INST_RECENT_PAY_DIFF   = ("PAY_DIFF", "mean"),
        ).reset_index()
        agg = agg.merge(recent_agg, on="SK_ID_CURR", how="left")
    return agg


def agg_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance -> 1 dong / khach hang."""
    agg = pos.groupby("SK_ID_CURR").agg(
        POS_COUNT            = ("SK_ID_PREV", "count"),
        POS_MONTHS_BAL_MEAN  = ("MONTHS_BALANCE", "mean"),
        POS_CNT_INST_MEAN    = ("CNT_INSTALMENT", "mean"),
        POS_DPD_MEAN         = ("SK_DPD", "mean"),
        POS_DPD_MAX          = ("SK_DPD", "max"),
        POS_DPD_DEF_MEAN     = ("SK_DPD_DEF", "mean"),
        POS_COMPLETED_RATIO  = ("NAME_CONTRACT_STATUS",
                                 lambda x: (x == "Completed").mean()),
        POS_ACTIVE_RATIO     = ("NAME_CONTRACT_STATUS",
                                 lambda x: (x == "Active").mean()),
    ).reset_index()
    return agg


def agg_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance -> 1 dong / khach hang."""
    agg = cc.groupby("SK_ID_CURR").agg(
        CC_COUNT             = ("SK_ID_PREV", "count"),
        CC_BALANCE_MEAN      = ("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX       = ("AMT_BALANCE", "max"),
        CC_LIMIT_MEAN        = ("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_DRAWINGS_MEAN     = ("AMT_DRAWINGS_CURRENT", "mean"),
        CC_DPD_MEAN          = ("SK_DPD", "mean"),
        CC_DPD_MAX           = ("SK_DPD", "max"),
    ).reset_index()

    agg["CC_UTILIZATION"] = (agg["CC_BALANCE_MEAN"] /
                              (agg["CC_LIMIT_MEAN"] + 1e-5))
    return agg


# ============================================================
# MERGE TAT CA
# ============================================================
def merge_all_tables(app_train: pd.DataFrame,
                     bureau: pd.DataFrame, bureau_bal: pd.DataFrame,
                     prev_app: pd.DataFrame, installments: pd.DataFrame,
                     pos_cash: pd.DataFrame, credit_card: pd.DataFrame
                     ) -> pd.DataFrame:
    """Aggregate va merge tat ca bang phu vao bang chinh."""
    print("[DataLoader] Aggregating auxiliary tables...")

    bur_agg  = agg_bureau(bureau, bureau_bal)
    prev_agg = agg_previous_application(prev_app)
    inst_agg = agg_installments(installments)
    pos_agg  = agg_pos_cash(pos_cash)
    cc_agg   = agg_credit_card(credit_card)

    n0 = app_train.shape[1]
    df = app_train.copy()
    for name, agg_df in [("bureau", bur_agg), ("previous", prev_agg),
                          ("installments", inst_agg), ("pos", pos_agg),
                          ("credit_card", cc_agg)]:
        df = df.merge(agg_df, on="SK_ID_CURR", how="left")
        print(f"  After merge {name:15s}: {df.shape[1]} columns")

    print(f"[DataLoader] Final shape: {df.shape} (+{df.shape[1]-n0} new features)")
    return df
