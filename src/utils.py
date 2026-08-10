"""
src/utils.py
============
Cac ham tien ich dung chung:
- Tinh metrics
- Visualization helpers
- Weighted blend ensemble
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from scipy.optimize import minimize
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from src.config import DARK_THEME, PALETTE, FIGURES_DIR


# ============================================================
# SETUP PLOT STYLE
# ============================================================
def setup_plot_style():
    """Ap dung dark theme cho matplotlib."""
    plt.rcParams.update(DARK_THEME)


def savefig(name: str, dpi: int = 150):
    """Luu figure vao thu muc figures/."""
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=DARK_THEME["figure.facecolor"])
    print(f"  [Plot] Saved: {path}")
    return path


# ============================================================
# METRICS
# ============================================================
def compute_all_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Tinh day du 6 metrics cho 1 model."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "AUC":           roc_auc_score(y_true, y_score),
        "Accuracy":      accuracy_score(y_true, y_pred),
        "Precision":     precision_score(y_true, y_pred, zero_division=0),
        "Recall":        recall_score(y_true, y_pred),
        "F1-Score":      f1_score(y_true, y_pred),
        "Avg Precision": average_precision_score(y_true, y_score),
    }


def metrics_table(preds_dict: dict, y_true, threshold: float = 0.5) -> pd.DataFrame:
    """Tao bang so sanh metrics giua nhieu models."""
    rows = {}
    for name, y_score in preds_dict.items():
        rows[name] = compute_all_metrics(y_true, y_score, threshold)
    df = pd.DataFrame(rows).T.round(4)
    df["Winner"] = "—"
    for col in ["AUC", "Accuracy", "Precision", "Recall", "F1-Score", "Avg Precision"]:
        if col in df.columns:
            df.loc[df[col].idxmax(), "Winner"] = col
    return df


# ============================================================
# WEIGHTED BLEND ENSEMBLE (Nelder-Mead)
# ============================================================
def neg_auc(weights, preds_list, y_true):
    """Ham muc tieu: -AUC (de minimize)."""
    w = np.clip(np.array(weights), 0, 1)
    w = w / w.sum()
    blend = sum(wi * pi for wi, pi in zip(w, preds_list))
    return -roc_auc_score(y_true, blend)


def optimize_blend(preds_list: list, y_true,
                    init_weights=None) -> tuple:
    """
    Tim trong so toi uu bang Nelder-Mead.
    Returns: (optimal_weights, blended_predictions, auc)
    """
    n = len(preds_list)
    if init_weights is None:
        init_weights = [1.0 / n] * n

    result = minimize(
        neg_auc, init_weights,
        args=(preds_list, y_true),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-7},
    )
    opt_w = np.clip(result.x, 0, 1)
    opt_w = opt_w / opt_w.sum()
    blend = sum(w * p for w, p in zip(opt_w, preds_list))
    auc   = roc_auc_score(y_true, blend)
    return opt_w, blend, auc


# ============================================================
# VISUALIZATION HELPERS
# ============================================================
def plot_auc_bars(aucs: dict, title: str = "AUC Comparison",
                  save_name: str = None, ax=None):
    """Ve bar chart so sanh AUC cac models."""
    color_order = [PALETTE["dae"], PALETTE["resnet"], PALETTE["wnd"],
                   PALETTE["ensemble"], PALETTE["lgbm"]]
    colors = color_order[:len(aucs)]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    names  = list(aucs.keys())
    values = list(aucs.values())
    bars = ax.bar(names, values, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3, width=0.6)
    ax.set_ylim(0.5, max(values) + 0.06)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, color="white", fontweight="bold")
    ax.set_title(title, color="white", fontsize=13)
    ax.set_ylabel("AUC-ROC", color="white")
    ax.set_facecolor(DARK_THEME["axes.facecolor"])
    ax.tick_params(axis="x", rotation=15)

    if standalone:
        plt.tight_layout()
        if save_name:
            savefig(save_name)
        plt.show()


def plot_roc_curves(preds_dict: dict, y_true, title: str = "ROC Curves",
                     save_name: str = None, ax=None):
    """Ve ROC curve cho nhieu models."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    color_list = list(PALETTE.values())
    for i, (name, y_score) in enumerate(preds_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, color=color_list[i % len(color_list)],
                lw=2, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "w--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title(title, color="white", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_facecolor(DARK_THEME["axes.facecolor"])

    if standalone:
        plt.tight_layout()
        if save_name:
            savefig(save_name)
        plt.show()


def plot_confusion_matrix(y_true, y_pred_proba, threshold: float = 0.5,
                            title: str = "Confusion Matrix",
                            save_name: str = None, ax=None):
    """Ve confusion matrix."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Default"],
                yticklabels=["Normal", "Default"],
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"{title}\nRecall={rec:.2%} | Precision={prec:.2%}",
                  color="white", fontsize=11)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.set_facecolor(DARK_THEME["axes.facecolor"])

    if standalone:
        plt.tight_layout()
        if save_name:
            savefig(save_name)
        plt.show()


def plot_radar_chart(metrics_dict: dict, title: str = "Radar Chart",
                      save_name: str = None):
    """
    Ve radar chart so sanh nhieu models tren nhieu metrics.
    metrics_dict: {"ModelName": {"AUC": 0.78, ...}}
    """
    metric_names = ["AUC", "Accuracy", "Precision", "Recall",
                    "F1-Score", "Avg Precision"]
    N      = len(metric_names)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_THEME["figure.facecolor"])
    ax.set_facecolor(DARK_THEME["axes.facecolor"])

    color_list = list(PALETTE.values())
    for i, (name, m) in enumerate(metrics_dict.items()):
        vals = [m.get(k, 0) for k in metric_names] + [m.get(metric_names[0], 0)]
        color = color_list[i % len(color_list)]
        ax.plot(angles, vals, "o-", color=color, lw=2.5,
                markersize=7, label=name)
        ax.fill(angles, vals, alpha=0.2, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=14,
                  fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(color=DARK_THEME["grid.color"], alpha=0.6)
    ax.tick_params(colors="white")

    plt.tight_layout()
    if save_name:
        savefig(save_name)
    plt.show()


def plot_training_time(times_dict: dict, title: str = "Training Time",
                        save_name: str = None):
    """Ve bar chart so sanh thoi gian training."""
    fig, ax = plt.subplots(figsize=(12, 5))
    color_list = list(PALETTE.values())
    items = list(times_dict.items())

    bars = ax.bar(
        [k for k, _ in items],
        [v / 60 for _, v in items],
        color=color_list[:len(items)],
        edgecolor="white", linewidth=1.2, zorder=3, width=0.5,
    )
    for bar, (_, secs) in zip(bars, items):
        mins = secs / 60
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{mins:.1f}m", ha="center", va="bottom",
                fontsize=12, color="white", fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", color="white")
    ax.set_ylabel("Thoi gian (phut)", color="white")
    ax.set_facecolor(DARK_THEME["axes.facecolor"])

    total_dl = sum(v for k, v in items if k != "LightGBM") / 60
    lgbm_t   = times_dict.get("LightGBM", 90) / 60
    ax.text(0.98, 0.95,
            f"DL Total: {total_dl:.1f}m\nLGBM: {lgbm_t:.1f}m",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#fdcb6e",
            bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.8))

    plt.tight_layout()
    if save_name:
        savefig(save_name)
    plt.show()
