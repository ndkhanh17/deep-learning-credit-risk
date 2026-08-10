# BÃ¡o CÃ¡o Äá»“ Ãn Nháº­p MÃ´n Há»c SÃ¢u â€“ NhÃ³m 11

## ThÃ´ng Tin NhÃ³m

| | |
|--|--|
| **MÃ´n há»c** | Nháº­p MÃ´n Há»c SÃ¢u |
| **GVHD** | TS. NgÃ´ Tiáº¿n Äá»©c |
| **NhÃ³m** | NhÃ³m 11 |
| **ThÃ nh viÃªn** | NgÃ´ HoÃ ng Nam Â· Nguyá»…n Duy KhÃ¡nh Â· Nguyá»…n Cao Chiáº¿n |
| **Dataset** | Home Credit Default Risk (Kaggle) |

---

## TÃ³m Táº¯t (Abstract)

Äá»“ Ã¡n nÃ y trÃ¬nh bÃ y há»‡ thá»‘ng Ä‘Ã¡nh giÃ¡ rá»§i ro tÃ­n dá»¥ng vÃ  phÃ¡t hiá»‡n há»“ sÆ¡ báº¥t thÆ°á»ng trÃªn táº­p dá»¯ liá»‡u **Home Credit Default Risk** vá»›i 307,511 há»“ sÆ¡ vay vÃ  7 báº£ng dá»¯ liá»‡u liÃªn quan. ChÃºng tÃ´i Ä‘á» xuáº¥t hai bÃ i toÃ¡n:

1. **BÃ i ToÃ¡n 1 â€“ Credit Risk Assessment** (Supervised Learning): XÃ¢y dá»±ng há»‡ thá»‘ng Ensemble gá»“m ba kiáº¿n trÃºc Deep Learning (DAE, Tabular ResNet, Wide & Deep) káº¿t há»£p vá»›i LightGBM baseline.

2. **BÃ i ToÃ¡n 2 â€“ Anomaly Detection** (Unsupervised Learning): PhÃ¡t hiá»‡n há»“ sÆ¡ báº¥t thÆ°á»ng sá»­ dá»¥ng Beta-VAE vá»›i há»‡ sá»‘ Î²=1.5 táº¡o khÃ´ng gian latent cÃ³ cáº¥u trÃºc.

---

## 1. Giá»›i Thiá»‡u & Má»¥c TiÃªu

### 1.1 Bá»‘i Cáº£nh

Home Credit lÃ  tá»• chá»©c tÃ i chÃ­nh phá»¥c vá»¥ nhÃ³m khÃ¡ch hÃ ng **underbanked** (thiáº¿u lá»‹ch sá»­ tÃ­n dá»¥ng). BÃ i toÃ¡n cá»‘t lÃµi lÃ  dá»± Ä‘oÃ¡n kháº£ nÄƒng vá»¡ ná»£ dá»±a trÃªn dá»¯ liá»‡u hÃ nh vi phi truyá»n thá»‘ng.

### 1.2 ThÃ¡ch Thá»©c

- **Máº¥t cÃ¢n báº±ng lá»›p nghiÃªm trá»ng**: 91.9% bÃ¬nh thÆ°á»ng vs 8.1% vá»¡ ná»£ (tá»· lá»‡ 11.4:1)
- **Dá»¯ liá»‡u nhiá»u chiá»u**: 7 báº£ng, tá»•ng ~56 triá»‡u hÃ ng
- **Missing values cao**: nhiá»u cá»™t missing >60%

### 1.3 Má»¥c TiÃªu

| BÃ i ToÃ¡n | Má»¥c TiÃªu ChÃ­nh | Metric |
|----------|----------------|--------|
| BT1 | PhÃ¢n loáº¡i rá»§i ro tÃ­n dá»¥ng | AUC-ROC |
| BT2 | PhÃ¡t hiá»‡n há»“ sÆ¡ báº¥t thÆ°á»ng | AUC-ROC, Precision, Recall |

---

## 2. Dá»¯ Liá»‡u

### 2.1 Thá»‘ng KÃª Tá»•ng Quan

| Báº£ng | Sá»‘ HÃ ng | Sá»‘ Cá»™t | MÃ´ Táº£ |
|------|---------|--------|-------|
| `application_train` | 307,511 | 122 | ÄÆ¡n vay (báº£ng chÃ­nh) |
| `application_test` | 48,744 | 121 | Táº­p test Kaggle |
| `bureau` | 1,716,428 | 17 | Lá»‹ch sá»­ tÃ­n dá»¥ng tá»« CIC |
| `bureau_balance` | 27,299,925 | 3 | Tráº¡ng thÃ¡i hÃ ng thÃ¡ng bureau |
| `previous_application` | 1,670,214 | 37 | ÄÆ¡n vay trÆ°á»›c táº¡i Home Credit |
| `installments_payments` | 13,605,401 | 8 | Lá»‹ch sá»­ tráº£ gÃ³p |
| `POS_CASH_balance` | 10,001,358 | 8 | Sá»‘ dÆ° POS & tiá»n máº·t |
| `credit_card_balance` | 3,840,312 | 23 | Sá»‘ dÆ° tháº» tÃ­n dá»¥ng |

### 2.2 PhÃ¢n Bá»‘ TARGET (HÃ¬nh 3.1)

![HÃ¬nh 3.1: PhÃ¢n Bá»‘ TARGET](../figures/fig_3_1_target_distribution.png)

**Nháº­n xÃ©t:** Máº¥t cÃ¢n báº±ng nghiÃªm trá»ng yÃªu cáº§u chiáº¿n lÆ°á»£c xá»­ lÃ½ Ä‘áº·c biá»‡t:
- Sá»­ dá»¥ng **Focal Loss** thay BCE thÃ´ng thÆ°á»ng
- **Class weighting**: `scale_pos_weight = 11.4`
- Metric chÃ­nh: **AUC-ROC** (khÃ´ng bá»‹ áº£nh hÆ°á»Ÿng bá»Ÿi máº¥t cÃ¢n báº±ng)

---

## 3. Feature Engineering

### 3.1 SÆ¡ Äá»“ Pipeline

```
Raw Data (7 báº£ng)
      â”‚
      â–¼
Aggregation Layer
  â”œâ”€â”€ agg_bureau()              â†’ 13 features
  â”œâ”€â”€ agg_previous_application() â†’ 8 features
  â”œâ”€â”€ agg_installments()        â†’ 8 features (+ recent 2)
  â”œâ”€â”€ agg_pos_cash()            â†’ 8 features
  â””â”€â”€ agg_credit_card()         â†’ 8 features + utilization
      â”‚
      â–¼
Feature Construction
  â”œâ”€â”€ Ratio Features            â†’ 9 features
  â”‚     CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, ...
  â”œâ”€â”€ EXT_SOURCE Interactions   â†’ ~25 features
  â”‚     Pairwise products, divisions, polynomials
  â””â”€â”€ Target Encoding           â†’ ~32 features (Bayesian smooth m=20)
      â”‚
      â–¼
Preprocessing
  â”œâ”€â”€ Label Encoding (categorical)
  â”œâ”€â”€ Drop missing > 60%
  â”œâ”€â”€ Median Imputation
  â””â”€â”€ QuantileTransformer â†’ N(0,1)
      â”‚
      â–¼
Final Feature Matrix: ~200-250 features
```

### 3.2 EXT_SOURCE â€“ NhÃ³m Features Quan Trá»ng Nháº¥t (HÃ¬nh 3.3)

![HÃ¬nh 3.3: EXT_SOURCE](../figures/fig_3_3_ext_source.png)

EXT_SOURCE_1/2/3 lÃ  Ä‘iá»ƒm tÃ­n dá»¥ng tá»« nguá»“n ngoÃ i (CIC, Ä‘á»‘i tÃ¡c). TÆ°Æ¡ng quan Ã¢m máº¡nh vá»›i TARGET:

| Feature | Correlation vá»›i TARGET | Missing |
|---------|----------------------|---------|
| EXT_SOURCE_1 | -0.155 | 56.4% |
| EXT_SOURCE_2 | -0.160 | 0.3% |
| EXT_SOURCE_3 | -0.178 | 19.8% |

### 3.3 Biáº¿n TÃ i ChÃ­nh (HÃ¬nh 3.4)

![HÃ¬nh 3.4: Boxplot](../figures/fig_3_4_boxplot.png)

### 3.4 Ma Tráº­n TÆ°Æ¡ng Quan (HÃ¬nh 3.5)

![HÃ¬nh 3.5: Correlation](../figures/fig_3_5_correlation.png)

### 3.5 Missing Values (HÃ¬nh 3.6)

![HÃ¬nh 3.6: Missing Values](../figures/fig_3_6_missing_values.png)

**Chiáº¿n lÆ°á»£c:** Drop cá»™t missing > 60%, median impute cÃ²n láº¡i.

### 3.6 QuantileTransformer (HÃ¬nh 3.7)

![HÃ¬nh 3.7: Quantile Transform](../figures/fig_3_7_quantile_transform.png)

Biáº¿n Ä‘á»•i phÃ¢n phá»‘i lá»‡ch (skewed) â†’ chuáº©n N(0,1). Lá»£i Ã­ch cho DL: gradient á»•n Ä‘á»‹nh hÆ¡n.

### 3.7 TÄƒng TrÆ°á»Ÿng Features (HÃ¬nh 3.9)

![HÃ¬nh 3.9: Feature Growth](../figures/fig_3_9_feature_growth.png)

---

## 4. BÃ i ToÃ¡n 1: ÄÃ¡nh GiÃ¡ Rá»§i Ro TÃ­n Dá»¥ng

### 4.1 Kiáº¿n TrÃºc CÃ¡c Model

#### Model A: DAE Classifier (Denoising AutoEncoder)

**2-Phase Training:**

```
PHASE 1 â€“ Pretrain (Unsupervised, 8 epochs):
  Input â†’ SwapNoise(15%) â†’ Encoder(512â†’256â†’128) â†’ Decoder(256â†’512) â†’ Reconstruct
  Loss: MSE(x, xÌ‚)

PHASE 2 â€“ Finetune (Supervised, 40 epochs):
  Input â†’ Encoder(frozen) â†’ Dropout(0.3) â†’ Dense(64) â†’ sigmoid
  Loss: Focal Loss (Î³=2.0, Î±=0.7)
```

**Æ¯u Ä‘iá»ƒm:** Há»c biá»ƒu diá»…n robust khÃ´ng cáº§n label (Phase 1), sau Ä‘Ã³ fine-tune classification (Phase 2).

#### Model B: Tabular ResNet

```
Input(N) â†’ Dense(256) â†’ Swish â†’ BN â†’ Dropout(0.3)
         â†’ ResBlock(256, dr=0.30)  â† skip connection
         â†’ ResBlock(128, dr=0.20)  â† skip connection
         â†’ ResBlock(64,  dr=0.15)  â† skip connection
         â†’ BN â†’ Dropout(0.1) â†’ Dense(1) â†’ sigmoid
```

**ResBlock:**
```
x â†’ BN â†’ Dense â†’ Swish â†’ Dropout â†’ BN â†’ Dense â†’ Swish â†’ Add(x, shortcut)
```

#### Model C: Wide & Deep

```
Input(N) â”€â”¬â”€ Wide:  Dense(1)              (memorization â€“ linear patterns)
           â””â”€ Deep:  Dense(256â†’128â†’64)     (generalization â€“ non-linear)
                          â†“
                    Concat(Wide, Deep) â†’ Dense(1) â†’ sigmoid
```

#### Ensemble: Weighted Blend (Nelder-Mead)

```
Å· = wâ‚Â·DAE + wâ‚‚Â·ResNet + wâ‚ƒÂ·W&D
```

Tá»‘i Æ°u `w` báº±ng Nelder-Mead minimizing `-AUC_OOF`.

### 4.2 Káº¿t Quáº£ BT1 (HÃ¬nh 4.1)

![HÃ¬nh 4.1: Káº¿t Quáº£ BT1](../figures/fig_4_1_results_6panels.png)

### 4.3 Báº£ng So SÃ¡nh Metrics

| Model | AUC-ROC | Precision | Recall | F1-Score | Avg Precision |
|-------|---------|-----------|--------|----------|---------------|
| LightGBM | **0.784** | 0.385 | 0.512 | 0.440 | 0.312 |
| DAE | 0.758 | 0.362 | 0.488 | 0.416 | 0.289 |
| ResNet | 0.763 | 0.371 | 0.495 | 0.424 | 0.295 |
| W&D | 0.761 | 0.368 | 0.491 | 0.421 | 0.293 |
| **DL Ensemble** | **0.773** | **0.379** | **0.502** | **0.432** | **0.304** |

> *Káº¿t quáº£ trÃªn full data 307K. Sample nhá» 10K cÃ³ AUC tháº¥p hÆ¡n ~3-5%.*

### 4.4 PhÃ¢n TÃ­ch

- **LightGBM** váº«n máº¡nh hÆ¡n DL Ensemble ~1.1% AUC â€“ phÃ¹ há»£p vá»›i tÃ i liá»‡u tham kháº£o (tabular data)
- **DAE** khai thÃ¡c unsupervised pretraining, há»c phÃ¢n phá»‘i dá»¯ liá»‡u tá»‘t hÆ¡n trÃªn raw features
- **Weighted Blend** cáº£i thiá»‡n +1.5% so vá»›i Ä‘Æ¡n láº» nhá» Ä‘a dáº¡ng hÃ³a lá»—i
- **Focal Loss** (Î³=2.0) giáº£m Ä‘Ã¡ng ká»ƒ false negative trÃªn minority class

---

## 5. BÃ i ToÃ¡n 2: PhÃ¡t Hiá»‡n Há»“ SÆ¡ Báº¥t ThÆ°á»ng (Beta-VAE)

### 5.1 Kiáº¿n TrÃºc Beta-VAE

```
ENCODER:
  x(N) â†’ Dense(512) â†’ Dense(256) â†’ Dense(128)
        â†’ z_mean(64), z_log_var(64)
        â†’ z = z_mean + exp(0.5Â·z_log_var)Â·Îµ   [Reparameterization]

DECODER:
  z(64) â†’ Dense(128) â†’ Dense(256) â†’ Dense(512) â†’ xÌ‚(N)

LOSS FUNCTION:
  L = ||x - xÌ‚||Â² + Î²Â·KL(q(z|x) || N(0,I))

  Trong Ä‘Ã³ Î² = 1.5 > 1:
  - Î² lá»›n â†’ khÃ´ng gian latent rÃµ rÃ ng, Ä‘á»™c láº­p hÆ¡n
  - Giáº£m false positive trong phÃ¡t hiá»‡n báº¥t thÆ°á»ng
```

### 5.2 CÆ¡ Cháº¿ PhÃ¡t Hiá»‡n Báº¥t ThÆ°á»ng

```
Training: chá»‰ dÃ¹ng dá»¯ liá»‡u bÃ¬nh thÆ°á»ng (TARGET=0)
          VAE há»c phÃ¢n phá»‘i P(x|z) cá»§a "khÃ¡ch hÃ ng khá»e máº¡nh"

Inference: anomaly_score(x) = ||x - Decoder(Encoder(x))||Â²
           threshold = percentile_95(scores[training])
           y_pred = 1 if score > threshold else 0
```

**Trá»±c giÃ¡c:** Náº¿u khÃ¡ch hÃ ng **báº¥t thÆ°á»ng** â†’ VAE khÃ´ng thá»ƒ tÃ¡i táº¡o tá»‘t â†’ lá»—i cao â†’ flagged as anomaly.

### 5.3 Káº¿t Quáº£ BT2 (HÃ¬nh 5.1)

![HÃ¬nh 5.1: VAE Results](../figures/fig_5_1_vae_results.png)

| Metric | GiÃ¡ Trá»‹ |
|--------|---------|
| AUC-ROC | 0.565 |
| Avg Precision | 0.143 |
| Precision@p95 | 0.182 |
| Recall@p95 | 0.502 |
| F1@p95 | 0.267 |

**PhÃ¢n tÃ­ch KL Collapse:** Î²=1.5 giÃºp trÃ¡nh KL collapse (nhiá»u dimension KLâ‰ˆ0), duy trÃ¬ latent space cÃ³ thÃ´ng tin.

---

## 6. So SÃ¡nh & Tá»•ng Káº¿t

### 6.1 Radar Chart (HÃ¬nh 7.2)

![HÃ¬nh 7.2: Radar Chart](../figures/fig_7_2_radar_chart.png)

### 6.2 Thá»i Gian Training (HÃ¬nh 7.3)

![HÃ¬nh 7.3: Training Time](../figures/fig_7_3_training_time.png)

| Model | Thá»i Gian (full data) |
|-------|----------------------|
| DAE (3-fold) | ~25 phÃºt |
| ResNet (3-fold) | ~18 phÃºt |
| W&D (3-fold) | ~20 phÃºt |
| LightGBM (3-fold) | ~8 phÃºt |
| Beta-VAE | ~12 phÃºt |

### 6.3 Káº¿t Luáº­n

| TiÃªu ChÃ­ | LightGBM | DL Ensemble | Beta-VAE |
|----------|----------|-------------|---------|
| AUC | **0.784** | 0.773 | 0.565 |
| Interpretability | Cao (SHAP) | Trung bÃ¬nh | Tháº¥p |
| Training Time | **Nhanh** | Cháº­m (3-5x) | Trung bÃ¬nh |
| Imbalance Handling | Good | **Focal Loss** | Unsupervised |
| Use Case | Production | Ensemble | Anomaly Detection |

### 6.4 HÆ°á»›ng PhÃ¡t Triá»ƒn

1. **FT-Transformer** (Feature Tokenization Transformer) â€“ attention trÃªn features báº£ng
2. **Graph Neural Network** â€“ mÃ´ hÃ¬nh hÃ³a quan há»‡ máº¡ng lÆ°á»›i khÃ¡ch hÃ ng
3. **Multi-task Learning** â€“ káº¿t há»£p BT1 vÃ  BT2 trong má»™t framework
4. **Online Learning** â€“ cáº­p nháº­t model theo drift dá»¯ liá»‡u thá»i gian thá»±c
5. **Explainability** â€“ SHAP values cho DL Ensemble

---

## 7. TÃ i Liá»‡u Tham Kháº£o

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
2. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
4. Cheng, H. T., et al. (2016). Wide & Deep Learning for Recommender Systems. *DLRS Workshop*.
5. Higgins, I., et al. (2017). Î²-VAE: Learning Basic Visual Concepts with a Constrained VAE. *ICLR*.
6. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
7. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *ICLR*.

---

*BÃ¡o cÃ¡o Ä‘Æ°á»£c táº¡o tá»± Ä‘á»™ng tá»« pipeline. Figures trong thÆ° má»¥c `../figures/`.*

