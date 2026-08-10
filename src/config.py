"""
src/config.py
=============
Cau hinh chung cho du an Home Credit Default Risk - Nhom 11
"""
from pathlib import Path

# ============================================================
# DUONG DAN
# ============================================================
ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "home-credit-default-risk"
PROCESSED_DIR  = ROOT_DIR / "data" / "processed"
FIGURES_DIR    = ROOT_DIR / "figures"
REPORTS_DIR    = ROOT_DIR / "reports"
NOTEBOOKS_DIR  = ROOT_DIR / "notebooks"

# Tao thu muc neu chua co
for d in [PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# CHE DO DU LIEU
# ============================================================
FULL_DATA   = False      # True = full 307K, False = sample
SAMPLE_SIZE = 10_000     # Chi dung khi FULL_DATA = False
MAX_ROWS_AUX = 200_000   # Gioi han hang bang phu (khi sample)

# ============================================================
# THAM SO CHUNG
# ============================================================
SEED    = 42
N_FOLDS = 3

# ============================================================
# LIGHTGBM HYPERPARAMETERS
# ============================================================
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     SEED,
    "n_jobs":           -1,
    "verbose":          -1,
}
LGBM_ROUNDS         = 500
LGBM_EARLY_STOPPING = 50

# ============================================================
# DEEP LEARNING HYPERPARAMETERS
# ============================================================
DL_BATCH_SIZE  = 2048
DL_EPOCHS      = 40
DL_PATIENCE    = 8
DL_LR          = 1e-3
DL_WEIGHT_DECAY= 1e-4

# DAE Pretrain
DAE_PRETRAIN_EPOCHS = 8
DAE_SWAP_NOISE      = 0.15

# Focal Loss
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.7

# Beta-VAE
VAE_LATENT_DIM   = 64
VAE_BETA         = 1.5
VAE_EPOCHS       = 20
VAE_BATCH_SIZE   = 512
VAE_LR           = 5e-4
VAE_THRESHOLD_PCT = 95   # percentile de xac dinh nguong bat thuong

# ============================================================
# FEATURE ENGINEERING
# ============================================================
TARGET_ENCODE_COLS = [
    "ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
]
TE_SMOOTH = 20          # Bayesian smoothing factor
MISSING_THRESHOLD = 0.60 # Loai bo cot co missing > 60%

# ============================================================
# VISUALIZATION
# ============================================================
DARK_THEME = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#0f3460",
    "axes.labelcolor":  "#e2e2e2",
    "text.color":       "#e2e2e2",
    "xtick.color":      "#e2e2e2",
    "ytick.color":      "#e2e2e2",
    "grid.color":       "#0f3460",
    "grid.alpha":       0.4,
    "axes.grid":        True,
    "font.size":        11,
}

PALETTE = {
    "normal":   "#00d2ff",
    "default":  "#ff6b6b",
    "ensemble": "#a29bfe",
    "lgbm":     "#fd79a8",
    "resnet":   "#55efc4",
    "wnd":      "#fdcb6e",
    "dae":      "#74b9ff",
    "vae":      "#81ecec",
}
