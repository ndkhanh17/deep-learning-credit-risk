"""
src/models.py
=============
Kien truc cac model Deep Learning:
  Model A: DAE Classifier (Denoising AutoEncoder)
  Model B: Tabular ResNet (Skip Connections)
  Model C: Wide & Deep
  Model D: Beta-VAE (Anomaly Detection)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from src.config import (
    DL_LR, DL_WEIGHT_DECAY, FOCAL_GAMMA, FOCAL_ALPHA,
    VAE_LATENT_DIM, VAE_BETA, VAE_LR,
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================
def focal_loss(gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
    """
    Focal Loss cho imbalanced classification:
      FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Giam anh huong cua easy samples, tap trung vao hard samples.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce    = (-y_true * tf.math.log(y_pred) -
                  (1 - y_true) * tf.math.log(1 - y_pred))
        p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal   = alpha_t * tf.pow(1 - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss_fn


# ============================================================
# CUSTOM LAYERS
# ============================================================
class SwapNoise(keras.layers.Layer):
    """
    Swap Noise: hoan doi ngau nhien p% gia tri trong batch.
    Dung trong pha Pretrain cua DAE.
    """
    def __init__(self, p: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, inputs, training=None):
        if not training:
            return inputs
        batch = tf.shape(inputs)[0]
        n_feat = tf.shape(inputs)[1]
        mask = tf.cast(tf.random.uniform((batch, n_feat)) < self.p, tf.float32)
        shuffle_idx = tf.random.shuffle(tf.range(batch))
        shuffled = tf.gather(inputs, shuffle_idx)
        return inputs * (1 - mask) + shuffled * mask

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p})
        return config


class Sampling(keras.layers.Layer):
    """
    Reparameterization trick cho VAE:
      z = mu + exp(0.5 * log_var) * epsilon
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ============================================================
# CALLBACKS
# ============================================================
class AUCEarlyStopping(keras.callbacks.Callback):
    """
    Custom callback: monitor AUC tren validation set,
    luu best weights va early stopping.
    """
    def __init__(self, X_val, y_val, patience: int = 8):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.best_auc = 0.0
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import roc_auc_score
        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        auc = roc_auc_score(self.y_val, y_pred)

        if auc > self.best_auc:
            self.best_auc = auc
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: val_auc={auc:.4f} "
                  f"(best={self.best_auc:.4f})")


# ============================================================
# MODEL A: DENOISING AUTOENCODER (DAE)
# ============================================================
def build_dae_pretrain(n_feat: int) -> keras.Model:
    """
    Phase 1 – Pretrain (Unsupervised):
    Input -> SwapNoise -> Encoder(512->256->128) -> Decoder(256->512) -> Reconstruct
    Loss: MSE
    """
    inp = keras.Input(shape=(n_feat,), name="dae_input")
    x   = SwapNoise(p=0.15, name="swap_noise")(inp)

    # Encoder
    x = layers.Dense(512, name="enc_dense_1")(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, name="enc_dense_2")(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    latent = layers.Dense(128, name="latent")(x)

    # Decoder
    x = layers.Dense(256)(latent)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512)(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    recon = layers.Dense(n_feat, name="reconstruction")(x)

    model = Model(inp, recon, name="DAE_Pretrain")
    model.compile(
        optimizer=keras.optimizers.AdamW(DL_LR, weight_decay=DL_WEIGHT_DECAY),
        loss="mse"
    )
    return model


def build_dae_finetune(pretrain_model: keras.Model, n_feat: int) -> keras.Model:
    """
    Phase 2 – Finetune (Supervised):
    Input -> Encoder(frozen) -> Dropout -> Dense(64) -> sigmoid
    Loss: Focal Loss
    
    NOTE: Keras 3.x khong ho tro weights= trong Dense constructor.
    Dung set_weights() sau khi model duoc build.
    """
    inp = keras.Input(shape=(n_feat,), name="dae_ft_input")

    # Encoder (cau truc, chua set weights)
    x = layers.Dense(512, trainable=False, name="enc_1_ft")(inp)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization(trainable=False)(x)
    x = layers.Dense(256, trainable=False, name="enc_2_ft")(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization(trainable=False)(x)
    x = layers.Dense(128, trainable=False, name="latent_ft")(x)

    # Classifier head (trainable)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64)(x)
    x   = layers.Activation("swish")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inp, out, name="DAE_Finetune")
    model.compile(
        optimizer=keras.optimizers.AdamW(DL_LR, weight_decay=DL_WEIGHT_DECAY),
        loss=focal_loss(FOCAL_GAMMA, FOCAL_ALPHA),
    )

    # Set pretrained weights SAU KHI build (Keras 3.x compatible)
    # Phai goi predict mot lan de build weights truoc
    import numpy as np
    _ = model(np.zeros((1, n_feat), dtype='float32'), training=False)
    model.get_layer("enc_1_ft").set_weights(
        pretrain_model.get_layer("enc_dense_1").get_weights())
    model.get_layer("enc_2_ft").set_weights(
        pretrain_model.get_layer("enc_dense_2").get_weights())
    model.get_layer("latent_ft").set_weights(
        pretrain_model.get_layer("latent").get_weights())
    return model


# ============================================================
# MODEL B: TABULAR RESNET
# ============================================================
def _res_block(x, units: int, dropout_rate: float, l2: float = 1e-5):
    """
    ResBlock: BN -> Dense -> Swish -> Dropout -> BN -> Dense -> Swish -> +skip
    Skip connection: giai quyet vanishing gradient
    """
    shortcut = x
    if x.shape[-1] != units:
        shortcut = layers.Dense(
            units, use_bias=False,
            kernel_regularizer=keras.regularizers.l2(l2)
        )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(units,
                      kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units,
                      kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = layers.Activation("swish")(x)
    return layers.Add()([x, shortcut])


def build_tabular_resnet(n_feat: int) -> keras.Model:
    """
    Tabular ResNet: 3 ResBlocks voi kích thuoc giam dan (256->128->64).
    Loss: Focal Loss
    """
    inp = keras.Input(shape=(n_feat,), name="resnet_input")

    # Initial projection
    x = layers.Dense(256)(inp)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual blocks
    x = _res_block(x, 256, dropout_rate=0.30)
    x = _res_block(x, 128, dropout_rate=0.20)
    x = _res_block(x, 64,  dropout_rate=0.15)

    # Head
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inp, out, name="TabularResNet")
    model.compile(
        optimizer=keras.optimizers.AdamW(2e-3, weight_decay=DL_WEIGHT_DECAY),
        loss=focal_loss(FOCAL_GAMMA, FOCAL_ALPHA),
    )
    return model


# ============================================================
# MODEL C: WIDE & DEEP
# ============================================================
def build_wide_and_deep(n_feat: int) -> keras.Model:
    """
    Wide & Deep Learning:
      Wide  (memorization) : Dense(1) - ghi nho pattern tuyen tinh
      Deep  (generalization): MLP(256->128->64)
      Concat(wide, deep) -> sigmoid
    """
    inp = keras.Input(shape=(n_feat,), name="wnd_input")

    # Wide branch
    wide = layers.Dense(1, name="wide_linear")(inp)

    # Deep branch
    x = layers.Dense(256)(inp)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128)(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64)(x)
    x = layers.Activation("swish")(x)
    deep = layers.Dropout(0.1)(x)

    # Combine
    combined = layers.Concatenate(name="wide_deep_concat")([wide, deep])
    out      = layers.Dense(1, activation="sigmoid", name="output")(combined)

    model = Model(inp, out, name="WideAndDeep")
    model.compile(
        optimizer=keras.optimizers.AdamW(DL_LR, weight_decay=DL_WEIGHT_DECAY),
        loss=focal_loss(FOCAL_GAMMA, FOCAL_ALPHA),
    )
    return model


# ============================================================
# MODEL D: BETA-VAE (ANOMALY DETECTION)
# ============================================================
def build_beta_vae(n_feat: int,
                    latent_dim: int = VAE_LATENT_DIM,
                    beta: float = VAE_BETA):
    """
    Beta-VAE cho anomaly detection:
      Encoder: n_feat -> 512->256->128 -> z_mean(64), z_log_var(64)
      Decoder: z(64)  -> 128->256->512 -> n_feat
      Loss: Recon + beta * KL

    NOTE: Keras 3.x yeu cau wrap tf.* ops trong Layer.
          Su dung BetaVAELoss custom layer + keras.ops thay vi tf.*

    Returns: (vae, encoder, decoder)
    """
    # --- ENCODER ---
    enc_inp = keras.Input(shape=(n_feat,), name="enc_input")
    x = layers.Dense(512)(enc_inp)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256)(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128)(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_sample  = Sampling(name="sampling")([z_mean, z_log_var])

    encoder = Model(enc_inp, [z_mean, z_log_var, z_sample], name="Encoder")

    # --- DECODER ---
    dec_inp = keras.Input(shape=(latent_dim,), name="dec_input")
    x = layers.Dense(128)(dec_inp)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256)(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512)(x)
    x = layers.Activation("swish")(x)
    x = layers.BatchNormalization()(x)
    dec_out = layers.Dense(n_feat, name="reconstruction")(x)

    decoder = Model(dec_inp, dec_out, name="Decoder")

    # --- VAE (full model) ---
    # Keras 3.x: wrap all ops inside a Layer subclass
    class BetaVAELoss(layers.Layer):
        """Tinh Beta-VAE loss (Recon + beta*KL) va add_loss."""
        def __init__(self, beta_val, **kwargs):
            super().__init__(**kwargs)
            self.beta_val = float(beta_val)

        def call(self, inputs):
            import keras.ops as ops
            x_orig, x_recon, z_m, z_lv = inputs
            recon = ops.mean(ops.square(x_orig - x_recon), axis=1)
            kl    = -0.5 * ops.mean(
                1.0 + z_lv - ops.square(z_m) - ops.exp(z_lv), axis=1
            )
            self.add_loss(ops.mean(recon + self.beta_val * kl))
            return x_recon   # pass-through

    vae_inp = keras.Input(shape=(n_feat,), name="vae_input")
    z_m, z_lv, z_s = encoder(vae_inp)
    x_hat   = decoder(z_s)
    vae_out = BetaVAELoss(beta_val=beta, name="beta_vae_loss")(
        [vae_inp, x_hat, z_m, z_lv]
    )

    vae = Model(vae_inp, vae_out, name="BetaVAE")
    vae.compile(
        optimizer=keras.optimizers.AdamW(VAE_LR, weight_decay=DL_WEIGHT_DECAY)
    )

    return vae, encoder, decoder
