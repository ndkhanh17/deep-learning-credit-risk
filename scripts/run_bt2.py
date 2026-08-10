"""
scripts/run_bt2.py
==================
Chay BT2: Beta-VAE Anomaly Detection + Summary
"""
import sys, os, warnings, pickle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')

import asyncio
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

print("Loading libraries...")
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from sklearn.decomposition import PCA

from src.config import *
from src.models import build_beta_vae
from src.utils import (setup_plot_style, savefig, metrics_table,
                        compute_all_metrics, plot_radar_chart, plot_training_time)

setup_plot_style()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"TF={tf.__version__}")

# ── Load features ──────────────────────────────────────────
X = np.load(str(PROCESSED_DIR / 'X.npy'))
y = np.load(str(PROCESSED_DIR / 'y.npy'))
print(f"X={X.shape} | Pos rate={y.mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════
# BT2: BETA-VAE
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("BT2: BETA-VAE ANOMALY DETECTION")
print("="*55)

X_normal  = X[y == 0].astype('float32')
X_default = X[y == 1].astype('float32')
print(f"Train (normal):  {len(X_normal):,} samples")
print(f"Eval (default):  {len(X_default):,} samples")
print(f"beta={VAE_BETA} | latent_dim={VAE_LATENT_DIM}")

n_feat = X.shape[1]
vae, encoder, decoder = build_beta_vae(n_feat, VAE_LATENT_DIM, VAE_BETA)

print(f"\nTraining Beta-VAE...")
t0 = time.time()
history = vae.fit(
    X_normal, X_normal,
    epochs=VAE_EPOCHS,
    batch_size=VAE_BATCH_SIZE,
    validation_split=0.1,
    verbose=0,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=0),
    ]
)
vae_time = time.time() - t0
n_epochs  = len(history.history['loss'])
print(f"Done: {vae_time:.0f}s | {n_epochs} epochs")
print(f"Train loss: {history.history['loss'][-1]:.4f}")
print(f"Val   loss: {history.history['val_loss'][-1]:.4f}")

# ── Anomaly Scores ─────────────────────────────────────────
print("\nComputing anomaly scores...")
X_all = X.astype('float32')
X_hat = vae.predict(X_all, batch_size=VAE_BATCH_SIZE, verbose=0)
recon_errors = np.mean((X_all - X_hat) ** 2, axis=1)

normal_errors  = recon_errors[y == 0]
default_errors = recon_errors[y == 1]
threshold = np.percentile(normal_errors, VAE_THRESHOLD_PCT)
y_pred    = (recon_errors >= threshold).astype(int)

tp = ((y_pred == 1) & (y == 1)).sum()
fp = ((y_pred == 1) & (y == 0)).sum()
fn = ((y_pred == 0) & (y == 1)).sum()
prec_v = tp / (tp + fp + 1e-10)
rec_v  = tp / (tp + fn + 1e-10)
f1_v   = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-10)
vae_auc = roc_auc_score(y, recon_errors)
vae_ap  = average_precision_score(y, recon_errors)

print(f"\nNormal  errors: mean={normal_errors.mean():.4f}, std={normal_errors.std():.4f}")
print(f"Default errors: mean={default_errors.mean():.4f}, std={default_errors.std():.4f}")
print(f"Threshold(p{VAE_THRESHOLD_PCT}): {threshold:.4f}")
print(f"\nAUC-ROC      : {vae_auc:.4f}")
print(f"Avg Precision: {vae_ap:.4f}")
print(f"Precision    : {prec_v:.4f}")
print(f"Recall       : {rec_v:.4f}")
print(f"F1-Score     : {f1_v:.4f}")

# ── Hình 5.1 – 6 panels ───────────────────────────────────
print("\nTao Hinh 5.1 (6 panels)...")
z_m, z_lv, _ = encoder.predict(X_all, batch_size=VAE_BATCH_SIZE, verbose=0)

fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# Panel 1: Training loss
ax1 = fig.add_subplot(gs[0, 0])
ep  = range(1, n_epochs + 1)
ax1.plot(ep, history.history['loss'],     color=PALETTE['vae'],     lw=2.5, label='Train')
ax1.plot(ep, history.history['val_loss'], color=PALETTE['default'], lw=2.5, ls='--', label='Val')
ax1.set_title('Training Loss (Recon + beta*KL)', color='white', fontsize=11)
ax1.set_xlabel('Epoch', color='white'); ax1.set_ylabel('Loss', color='white')
ax1.legend(fontsize=9); ax1.set_facecolor('#16213e')

# Panel 2: Error distribution
ax2 = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, np.percentile(recon_errors, 99), 60)
ax2.hist(normal_errors,  bins=bins, alpha=0.7, color=PALETTE['normal'],
         label=f'Normal ({len(normal_errors):,})', density=True)
ax2.hist(default_errors, bins=bins, alpha=0.7, color=PALETTE['default'],
         label=f'Default ({len(default_errors):,})', density=True)
ax2.axvline(threshold, color='#fdcb6e', ls='--', lw=2, label=f'Threshold (p{VAE_THRESHOLD_PCT})')
ax2.set_title(f'Reconstruction Error\nNormal={normal_errors.mean():.3f} | Default={default_errors.mean():.3f}',
              color='white', fontsize=10)
ax2.set_xlabel('Error', color='white'); ax2.set_ylabel('Density', color='white')
ax2.legend(fontsize=8); ax2.set_facecolor('#16213e')

# Panel 3: ROC
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y, recon_errors)
ax3.plot(fpr, tpr, color=PALETTE['vae'], lw=2.5, label=f'Beta-VAE (AUC={vae_auc:.4f})')
ax3.plot([0,1],[0,1], 'w--', alpha=0.4, lw=1)
ax3.fill_between(fpr, tpr, alpha=0.15, color=PALETTE['vae'])
ax3.set_title('ROC Curve – Anomaly Detection', color='white', fontsize=11)
ax3.set_xlabel('FPR', color='white'); ax3.set_ylabel('TPR', color='white')
ax3.legend(fontsize=10); ax3.set_facecolor('#16213e')

# Panel 4: Confusion matrix
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Normal', 'Default'], yticklabels=['Normal', 'Default'],
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 14})
ax4.set_title(f'Confusion Matrix (p{VAE_THRESHOLD_PCT})\nPrec={prec_v:.3f} | Rec={rec_v:.3f} | F1={f1_v:.3f}',
              color='white', fontsize=10)
ax4.set_xlabel('Predicted', color='white'); ax4.set_ylabel('Actual', color='white')
ax4.set_facecolor('#16213e')

# Panel 5: Latent space PCA
ax5 = fig.add_subplot(gs[1, 1])
pca = PCA(n_components=2, random_state=SEED)
z2d = pca.fit_transform(z_m)
ax5.scatter(z2d[y==0, 0], z2d[y==0, 1], s=4,  alpha=0.3, color=PALETTE['normal'],  label='Normal')
ax5.scatter(z2d[y==1, 0], z2d[y==1, 1], s=10, alpha=0.6, color=PALETTE['default'], label='Default')
ax5.set_title('Latent Space (PCA 2D)\nBeta-VAE Representation', color='white', fontsize=11)
ax5.set_xlabel('PC1', color='white'); ax5.set_ylabel('PC2', color='white')
ax5.legend(fontsize=9, markerscale=3); ax5.set_facecolor('#16213e')

# Panel 6: KL per dimension
ax6 = fig.add_subplot(gs[1, 2])
kl = np.mean(-0.5 * (1 + z_lv - z_m**2 - np.exp(z_lv)), axis=0)
top_idx = np.argsort(kl)[::-1][:15]
ax6.bar(range(15), kl[top_idx], color=PALETTE['vae'], edgecolor='white', linewidth=0.7)
ax6.set_title(f'Top 15 Latent Dims by KL (beta={VAE_BETA})', color='white', fontsize=11)
ax6.set_xlabel('Dimension', color='white'); ax6.set_ylabel('KL Divergence', color='white')
ax6.set_facecolor('#16213e')
kl_col = (kl < 0.01).sum()
ax6.text(0.98, 0.95, f'KL~0: {kl_col}/{VAE_LATENT_DIM}',
         transform=ax6.transAxes, ha='right', va='top', fontsize=10, color='#fdcb6e',
         bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))

fig.suptitle('Hinh 5.1: Beta-VAE Anomaly Detection – BT2',
             fontsize=14, fontweight='bold', color='white', y=1.01)
savefig('fig_5_1_vae_results.png')
print(f"  Saved fig_5_1_vae_results.png")
plt.close('all')

# Save
vae_res = {
    'recon_errors': recon_errors, 'y': y, 'threshold': threshold,
    'auc': vae_auc, 'ap': vae_ap,
    'precision': prec_v, 'recall': rec_v, 'f1': f1_v,
    'time': vae_time
}
with open(PROCESSED_DIR / 'bt2_vae_results.pkl', 'wb') as f:
    pickle.dump(vae_res, f)
print("  Saved bt2_vae_results.pkl")


# ═══════════════════════════════════════════════════════════
# SUMMARY: RADAR + TIME CHART
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("SUMMARY: TAO CAC HINH TONG HOP")
print("="*55)

with open(PROCESSED_DIR / 'bt1_predictions.pkl', 'rb') as f:
    bt1 = pickle.load(f)
with open(PROCESSED_DIR / 'training_times.pkl', 'rb') as f:
    times = pickle.load(f)

y_bt1 = bt1['y']

# Radar Chart
mc = {
    'DL Ensemble': compute_all_metrics(y_bt1, bt1['ensemble']),
    'LightGBM':    compute_all_metrics(y_bt1, bt1['lgbm']),
    'ResNet':      compute_all_metrics(y_bt1, bt1['resnet']),
    'W&D':         compute_all_metrics(y_bt1, bt1['wnd']),
}
plot_radar_chart(mc, title='Hinh 7.2: So Sanh Model – Radar Chart', save_name='fig_7_2_radar_chart.png')
print("  Saved fig_7_2_radar_chart.png")

# Training time
times['Beta-VAE'] = vae_time
plot_training_time(times, title='Hinh 7.3: Thoi Gian Training', save_name='fig_7_3_training_time.png')
print("  Saved fig_7_3_training_time.png")

# Summary figure (3 panels)
df_m = metrics_table({
    'LightGBM':   bt1['lgbm'],
    'DAE':        bt1['dae'],
    'ResNet':     bt1['resnet'],
    'W&D':        bt1['wnd'],
    'DL Ensemble': bt1['ensemble'],
}, y_bt1)

fig = plt.figure(figsize=(20, 8))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

all_aucs = {n: roc_auc_score(y_bt1, bt1[k])
            for n, k in [('LGBM','lgbm'),('DAE','dae'),('ResNet','resnet'),
                          ('W&D','wnd'),('Ensemble','ensemble')]}
all_aucs['Beta-VAE'] = vae_auc
c6 = [PALETTE['lgbm'], PALETTE['dae'], PALETTE['resnet'],
      PALETTE['wnd'], PALETTE['ensemble'], PALETTE['vae']]

ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(all_aucs.keys(), all_aucs.values(),
               color=c6, edgecolor='white', width=0.55, zorder=3)
for bar, val in zip(bars, all_aucs.values()):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom',
             fontsize=9, color='white', fontweight='bold')
ax1.set_ylim(0.45, max(all_aucs.values()) + 0.06)
ax1.axvline(4.5, color='white', ls='--', alpha=0.5)
ax1.text(0.37, 0.99, 'BT1', transform=ax1.transAxes,
         ha='center', va='top', fontsize=11, color='#fdcb6e')
ax1.text(0.93, 0.99, 'BT2', transform=ax1.transAxes,
         ha='center', va='top', fontsize=11, color='#81ecec')
ax1.set_title('AUC – BT1 & BT2', color='white', fontsize=11)
ax1.set_facecolor('#16213e'); ax1.tick_params(axis='x', rotation=25)

ax2 = fig.add_subplot(gs[0, 1])
for (name, key), c in zip([('LightGBM','lgbm'),('DL Ensemble','ensemble')],
                            [PALETTE['lgbm'], PALETTE['ensemble']]):
    fpr_, tpr_, _ = roc_curve(y_bt1, bt1[key])
    ax2.plot(fpr_, tpr_, color=c, lw=2.5, label=f'{name} ({all_aucs.get(name, all_aucs.get("Ensemble","")):.4f})')
ax2.plot(fpr, tpr, color=PALETTE['vae'], lw=2.5, ls='--', label=f'Beta-VAE ({vae_auc:.4f})')
ax2.plot([0,1],[0,1], 'w--', alpha=0.3, lw=1)
ax2.set_title('ROC – BT1 vs BT2', color='white', fontsize=11)
ax2.set_xlabel('FPR', color='white'); ax2.set_ylabel('TPR', color='white')
ax2.legend(fontsize=9); ax2.set_facecolor('#16213e')

ax3 = fig.add_subplot(gs[0, 2])
mn  = ['AUC', 'Precision', 'Recall', 'F1-Score']
ev  = [df_m.loc['DL Ensemble', m] for m in mn]
lv  = [df_m.loc['LightGBM',    m] for m in mn]
xp  = np.arange(len(mn)); w = 0.35
ax3.bar(xp - w/2, ev, w, label='DL Ensemble', color=PALETTE['ensemble'], edgecolor='white')
ax3.bar(xp + w/2, lv, w, label='LightGBM',    color=PALETTE['lgbm'],     edgecolor='white')
ax3.set_xticks(xp); ax3.set_xticklabels(mn, fontsize=9)
ax3.set_title('DL Ensemble vs LightGBM', color='white', fontsize=11)
ax3.set_ylim(0, 1.1); ax3.legend(fontsize=9); ax3.set_facecolor('#16213e')

fig.suptitle('Tong Ket – Nhan Mon Hoc Sau – Nhom 11',
             fontsize=15, fontweight='bold', color='white')
plt.tight_layout()
savefig('fig_tong_ket.png')
print("  Saved fig_tong_ket.png")
plt.close('all')


# ═══════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════
print()
print("=" * 55)
print("DU AN HOAN THANH TOAN BO!")
print("=" * 55)
print()
print("KET QUA BT1 (Credit Risk Assessment):")
for name, key in [('LightGBM','lgbm'),('DAE','dae'),('ResNet','resnet'),
                   ('W&D','wnd'),('DL Ensemble','ensemble')]:
    auc = roc_auc_score(y_bt1, bt1[key])
    flag = " <<< BEST" if auc == max(roc_auc_score(y_bt1, bt1[k])
                                       for k in ['lgbm','dae','resnet','wnd','ensemble']) else ""
    print(f"  {name:15s}: AUC={auc:.4f}{flag}")

print()
print("KET QUA BT2 (Anomaly Detection – Beta-VAE):")
print(f"  AUC-ROC      : {vae_auc:.4f}")
print(f"  Avg Precision: {vae_ap:.4f}")
print(f"  Precision@p95: {prec_v:.4f}")
print(f"  Recall@p95   : {rec_v:.4f}")
print(f"  F1@p95       : {f1_v:.4f}")

print()
figs = list(FIGURES_DIR.glob('*.png'))
print(f"Figures da tao ({len(figs)} files):")
for f in sorted(figs):
    print(f"  {f.name}")

total = sum(times.values())
print(f"\nTong thoi gian training: {total/60:.1f} phut")
print()
print("Mo bao cao: reports/BAO_CAO_NHOM11.md")
print("Mo JupyterLab: python -m jupyter lab")
