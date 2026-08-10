"""
scripts/run_bt1.py
==================
Chay truc tiep pipeline BT1 (khong qua nbconvert).
Thu chua loi va debug de hon.
"""
import sys, os, warnings, pickle, time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')

import asyncio
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

print("Loading libraries...")
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import lightgbm as lgb

from src.config import *
from src.models import (build_dae_pretrain, build_dae_finetune,
                         build_tabular_resnet, build_wide_and_deep)
from src.utils import (setup_plot_style, savefig, metrics_table,
                        optimize_blend, compute_all_metrics)

setup_plot_style()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"TF={tf.__version__} | LGB={lgb.__version__}")

# ── Load Data ──────────────────────────────────────────────
print("\nLoading X, y ...")
X = np.load(str(PROCESSED_DIR / 'X.npy'))
y = np.load(str(PROCESSED_DIR / 'y.npy'))
print(f"X={X.shape}  Pos rate={y.mean()*100:.1f}%")
scale_pos = (y == 0).sum() / (y == 1).sum()
print(f"scale_pos_weight = {scale_pos:.2f}")

# ── Custom AUC Early Stopping ──────────────────────────────
class AUCStop(keras.callbacks.Callback):
    def __init__(self, Xv, yv, pat=8):
        super().__init__()
        self.Xv, self.yv = Xv, yv
        self.pat = pat
        self.best = 0.0
        self.bw = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.Xv, verbose=0).flatten()
        auc  = roc_auc_score(self.yv, pred)
        if auc > self.best:
            self.best = auc
            self.bw   = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.pat:
                self.model.stop_training = True
        if (epoch + 1) % 5 == 0:
            print(f"    ep{epoch+1:3d}: val_auc={auc:.4f} (best={self.best:.4f})")


n_feat = X.shape[1]
kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

# ═══════════════════════════════════════════════════════════
# 1. LIGHTGBM
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("1. LIGHTGBM BASELINE")
print("="*55)
oof_lgbm   = np.zeros(len(y))
times_lgbm = []
lgbm_trees = []

for fold, (tr, va) in enumerate(kf.split(X, y)):
    t0 = time.time()
    sw = np.where(y[tr] == 1, scale_pos, 1.0)
    dtrain = lgb.Dataset(X[tr], y[tr], weight=sw)
    dval   = lgb.Dataset(X[va], y[va])
    m = lgb.train(
        LGBM_PARAMS, dtrain, LGBM_ROUNDS,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(-1),
        ]
    )
    oof_lgbm[va] = m.predict(X[va])
    auc = roc_auc_score(y[va], oof_lgbm[va])
    t1  = time.time() - t0
    times_lgbm.append(t1)
    lgbm_trees.append(m.num_trees())
    print(f"  Fold {fold+1}: AUC={auc:.4f} | trees={m.num_trees()} | {t1:.0f}s")
    del m

lgbm_auc = roc_auc_score(y, oof_lgbm)
print(f"  OOF AUC: {lgbm_auc:.4f}  (avg {np.mean(times_lgbm):.0f}s/fold)")

# ═══════════════════════════════════════════════════════════
# 2. DAE CLASSIFIER
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("2. DAE CLASSIFIER (2-PHASE)")
print("="*55)
oof_dae   = np.zeros(len(y))
times_dae = []

for fold, (tr, va) in enumerate(kf.split(X, y)):
    t0 = time.time()
    Xtr, Xva = X[tr].astype('float32'), X[va].astype('float32')
    ytr, yva = y[tr], y[va]
    sw = np.where(ytr == 1, scale_pos, 1.0)

    # Phase 1: Pretrain
    dae_pre = build_dae_pretrain(n_feat)
    dae_pre.fit(Xtr, Xtr,
                epochs=DAE_PRETRAIN_EPOCHS,
                batch_size=DL_BATCH_SIZE,
                validation_data=(Xva, Xva),
                verbose=0)
    mse = dae_pre.evaluate(Xva, Xva, verbose=0)
    print(f"  Fold {fold+1} Pretrain MSE={mse:.4f}")

    # Phase 2: Finetune
    dae_ft = build_dae_finetune(dae_pre, n_feat)
    cb = AUCStop(Xva, yva, DL_PATIENCE)
    dae_ft.fit(Xtr, ytr,
               sample_weight=sw,
               epochs=DL_EPOCHS,
               batch_size=DL_BATCH_SIZE,
               validation_data=(Xva, yva),
               verbose=0,
               callbacks=[cb])
    if cb.bw:
        dae_ft.set_weights(cb.bw)

    oof_dae[va] = dae_ft.predict(Xva, verbose=0).flatten()
    auc = roc_auc_score(yva, oof_dae[va])
    t1  = time.time() - t0
    times_dae.append(t1)
    print(f"  Fold {fold+1}: AUC={auc:.4f} | best_val={cb.best:.4f} | {t1:.0f}s")
    keras.backend.clear_session()
    del dae_pre, dae_ft

dae_auc = roc_auc_score(y, oof_dae)
print(f"  OOF AUC: {dae_auc:.4f}")

# ═══════════════════════════════════════════════════════════
# 3. TABULAR RESNET
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("3. TABULAR RESNET (SKIP CONNECTIONS)")
print("="*55)
oof_resnet   = np.zeros(len(y))
times_resnet = []

for fold, (tr, va) in enumerate(kf.split(X, y)):
    t0 = time.time()
    Xtr, Xva = X[tr].astype('float32'), X[va].astype('float32')
    ytr, yva = y[tr], y[va]
    sw = np.where(ytr == 1, scale_pos, 1.0)
    resnet = build_tabular_resnet(n_feat)
    cb = AUCStop(Xva, yva, DL_PATIENCE)
    resnet.fit(Xtr, ytr,
               sample_weight=sw,
               epochs=DL_EPOCHS,
               batch_size=DL_BATCH_SIZE,
               validation_data=(Xva, yva),
               verbose=0,
               callbacks=[cb])
    if cb.bw:
        resnet.set_weights(cb.bw)
    oof_resnet[va] = resnet.predict(Xva, verbose=0).flatten()
    auc = roc_auc_score(yva, oof_resnet[va])
    t1  = time.time() - t0
    times_resnet.append(t1)
    print(f"  Fold {fold+1}: AUC={auc:.4f} | best={cb.best:.4f} | {t1:.0f}s")
    keras.backend.clear_session()
    del resnet

resnet_auc = roc_auc_score(y, oof_resnet)
print(f"  OOF AUC: {resnet_auc:.4f}")

# ═══════════════════════════════════════════════════════════
# 4. WIDE & DEEP
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("4. WIDE & DEEP")
print("="*55)
oof_wnd   = np.zeros(len(y))
times_wnd = []

for fold, (tr, va) in enumerate(kf.split(X, y)):
    t0 = time.time()
    Xtr, Xva = X[tr].astype('float32'), X[va].astype('float32')
    ytr, yva = y[tr], y[va]
    sw = np.where(ytr == 1, scale_pos, 1.0)
    wnd = build_wide_and_deep(n_feat)
    cb = AUCStop(Xva, yva, DL_PATIENCE)
    wnd.fit(Xtr, ytr,
            sample_weight=sw,
            epochs=DL_EPOCHS,
            batch_size=DL_BATCH_SIZE,
            validation_data=(Xva, yva),
            verbose=0,
            callbacks=[cb])
    if cb.bw:
        wnd.set_weights(cb.bw)
    oof_wnd[va] = wnd.predict(Xva, verbose=0).flatten()
    auc = roc_auc_score(yva, oof_wnd[va])
    t1  = time.time() - t0
    times_wnd.append(t1)
    print(f"  Fold {fold+1}: AUC={auc:.4f} | best={cb.best:.4f} | {t1:.0f}s")
    keras.backend.clear_session()
    del wnd

wnd_auc = roc_auc_score(y, oof_wnd)
print(f"  OOF AUC: {wnd_auc:.4f}")

# ═══════════════════════════════════════════════════════════
# 5. ENSEMBLE (NELDER-MEAD)
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("5. WEIGHTED BLEND ENSEMBLE")
print("="*55)
preds_list  = [oof_dae, oof_resnet, oof_wnd]
model_names = ['DAE', 'ResNet', 'W&D']
opt_w, oof_ensemble, ens_auc = optimize_blend(preds_list, y)
print("Optimal weights:")
for name, w in zip(model_names, opt_w):
    print(f"  {name}: {w:.4f} ({w*100:.1f}%)")
print(f"  Ensemble AUC: {ens_auc:.4f}")

# ═══════════════════════════════════════════════════════════
# METRICS TABLE
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("BANG SO SANH METRICS (BT1)")
print("="*55)
all_preds = {
    'LightGBM':   oof_lgbm,
    'DAE':        oof_dae,
    'ResNet':     oof_resnet,
    'W&D':        oof_wnd,
    'DL Ensemble': oof_ensemble,
}
df_m = metrics_table(all_preds, y)
print(df_m[['AUC', 'Precision', 'Recall', 'F1-Score', 'Avg Precision']].to_string())

# ═══════════════════════════════════════════════════════════
# HINH 4.1 – 6 PANELS
# ═══════════════════════════════════════════════════════════
print("\nTao Hinh 4.1...")
aucs   = {k: roc_auc_score(y, v) for k, v in all_preds.items()}
colors5 = [PALETTE['lgbm'], PALETTE['dae'], PALETTE['resnet'],
           PALETTE['wnd'],  PALETTE['ensemble']]

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# Panel 1 – AUC bars
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(list(aucs.keys()), list(aucs.values()),
               color=colors5, edgecolor='white', width=0.6, zorder=3)
for bar, val in zip(bars, aucs.values()):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom',
             fontsize=9, color='white', fontweight='bold')
ax1.set_ylim(0.5, max(aucs.values()) + 0.05)
ax1.set_title('AUC-ROC Comparison', color='white', fontsize=11)
ax1.set_facecolor('#16213e')
ax1.tick_params(axis='x', rotation=25)

# Panel 2 – ROC curves
ax2 = fig.add_subplot(gs[0, 1])
for (name, y_sc), c in zip(all_preds.items(), colors5):
    fpr, tpr, _ = roc_curve(y, y_sc)
    lw = 2.5 if name == 'DL Ensemble' else 1.5
    ax2.plot(fpr, tpr, color=c, lw=lw, label=f'{name} ({aucs[name]:.4f})')
ax2.plot([0,1],[0,1], 'w--', alpha=0.4, lw=1)
ax2.set_xlabel('FPR', color='white'); ax2.set_ylabel('TPR', color='white')
ax2.set_title('ROC Curves', color='white', fontsize=11)
ax2.legend(fontsize=7, loc='lower right')
ax2.set_facecolor('#16213e')

# Panel 3 – PR curves
ax3 = fig.add_subplot(gs[0, 2])
for (name, y_sc), c in zip(all_preds.items(), colors5):
    prec_, rec_, _ = precision_recall_curve(y, y_sc)
    lw = 2.5 if name == 'DL Ensemble' else 1.5
    ax3.plot(rec_, prec_, color=c, lw=lw, label=name)
ax3.set_xlabel('Recall', color='white'); ax3.set_ylabel('Precision', color='white')
ax3.set_title('Precision-Recall Curves', color='white', fontsize=11)
ax3.legend(fontsize=7)
ax3.set_facecolor('#16213e')

# Panel 4 – Blend weights pie
ax4 = fig.add_subplot(gs[1, 0])
ax4.pie(opt_w, labels=model_names,
        colors=[PALETTE['dae'], PALETTE['resnet'], PALETTE['wnd']],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 12, 'color': 'white'},
        wedgeprops={'edgecolor': '#1a1a2e', 'linewidth': 2})
ax4.set_title(f'Optimal Blend Weights (AUC={ens_auc:.4f})',
              color='white', fontsize=11)
ax4.set_facecolor('#16213e')

# Panel 5 – Confusion matrix
ax5 = fig.add_subplot(gs[1, 1])
ens_pred = (oof_ensemble >= 0.5).astype(int)
cm = confusion_matrix(y, ens_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['Normal', 'Default'],
            yticklabels=['Normal', 'Default'],
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 14})
ax5.set_title('Confusion Matrix – DL Ensemble', color='white', fontsize=11)
ax5.set_xlabel('Predicted', color='white'); ax5.set_ylabel('Actual', color='white')
ax5.set_facecolor('#16213e')

# Panel 6 – Score distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(oof_ensemble[y == 0], bins=50, alpha=0.7,
         color=PALETTE['normal'], label='Normal', density=True)
ax6.hist(oof_ensemble[y == 1], bins=50, alpha=0.7,
         color=PALETTE['default'], label='Default', density=True)
ax6.set_title('Score Distribution – DL Ensemble', color='white', fontsize=11)
ax6.set_xlabel('Predicted Score', color='white')
ax6.set_ylabel('Density', color='white')
ax6.legend(fontsize=9)
ax6.set_facecolor('#16213e')

fig.suptitle('Hinh 4.1: Tong Ket BT1 – Credit Risk Assessment',
             fontsize=14, fontweight='bold', color='white', y=1.01)
savefig('fig_4_1_results_6panels.png')
print("  Saved fig_4_1_results_6panels.png")
plt.close('all')

# ═══════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════
preds_out = {
    'lgbm': oof_lgbm, 'dae': oof_dae, 'resnet': oof_resnet,
    'wnd': oof_wnd, 'ensemble': oof_ensemble, 'y': y
}
with open(PROCESSED_DIR / 'bt1_predictions.pkl', 'wb') as f:
    pickle.dump(preds_out, f)

times_out = {
    'DAE': sum(times_dae), 'ResNet': sum(times_resnet),
    'W&D': sum(times_wnd), 'LightGBM': sum(times_lgbm)
}
with open(PROCESSED_DIR / 'training_times.pkl', 'wb') as f:
    pickle.dump(times_out, f)

print()
print("="*55)
print("BT1 HOAN THANH!")
print("="*55)
for name, auc in aucs.items():
    marker = " <-- BEST DL" if name == "DL Ensemble" else (" <-- BEST" if auc == max(aucs.values()) else "")
    print(f"  {name:15s}: AUC={auc:.4f}{marker}")
print()
total_time = sum(times_lgbm) + sum(times_dae) + sum(times_resnet) + sum(times_wnd)
print(f"Total training time: {total_time/60:.1f} minutes")
print()
print("Chay tiep: python scripts/run_bt2.py")
