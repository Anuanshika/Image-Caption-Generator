"""Model: frozen EfficientNetB0 image encoder + Transformer caption decoder."""

from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops


# ---------------------------------------------------------------- CNN encoder
def build_cnn_encoder(img_size: int = 224, pretrained: bool = True) -> keras.Model:
    """Image (0-255 RGB) -> grid of feature vectors, shape (49, 1280) for 224 px.

    Each of the 7x7 = 49 vectors describes one region of the image; the
    decoder attends to these regions while writing each word.
    """
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = base(inputs, training=False)
    x = layers.Reshape((-1, x.shape[-1]), name="regions")(x)
    return keras.Model(inputs, x, name="cnn_encoder")


# ---------------------------------------------------------------- building blocks
class TokenAndPositionEmbedding(layers.Layer):
    """Word embedding + learned position embedding."""

    def __init__(self, vocab_size: int, max_len: int, d_model: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_len, d_model)
        self.scale = float(d_model) ** 0.5

    def call(self, tokens):
        positions = ops.arange(0, ops.shape(tokens)[-1])
        return self.token_emb(tokens) * self.scale + self.pos_emb(positions)


class FeedForward(layers.Layer):
    def __init__(self, d_model: int, ff_dim: int, dropout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(ff_dim, activation="relu")
        self.dense2 = layers.Dense(d_model)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        return self.drop(self.dense2(self.dense1(x)), training=training)


class ImageEncoder(layers.Layer):
    """Projects CNN regions to d_model and lets regions share context."""

    def __init__(self, d_model, num_heads, ff_dim, dropout, **kwargs) -> None:
        super().__init__(**kwargs)
        self.proj = layers.Dense(d_model, activation="relu")
        self.norm0 = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(num_heads, d_model // num_heads, dropout=dropout)
        self.norm1 = layers.LayerNormalization()
        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.norm2 = layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.norm0(self.proj(x))
        x = self.norm1(x + self.attn(x, x, training=training))
        return self.norm2(x + self.ffn(x, training=training))


class DecoderBlock(layers.Layer):
    """Masked self-attention (on words) + cross-attention (words -> image regions)."""

    def __init__(self, d_model, num_heads, ff_dim, dropout, **kwargs) -> None:
        super().__init__(**kwargs)
        key_dim = d_model // num_heads
        self.self_attn = layers.MultiHeadAttention(num_heads, key_dim, dropout=dropout)
        self.cross_attn = layers.MultiHeadAttention(num_heads, key_dim, dropout=dropout)
        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        self.drop = layers.Dropout(dropout)

    def call(self, x, image, training=False):
        # Each word can only look at earlier words (causal mask).
        a = self.self_attn(x, x, use_causal_mask=True, training=training)
        x = self.norm1(x + self.drop(a, training=training))
        # Each word looks at the image regions; scores = where it looked.
        c, scores = self.cross_attn(x, image, return_attention_scores=True, training=training)
        x = self.norm2(x + self.drop(c, training=training))
        x = self.norm3(x + self.ffn(x, training=training))
        return x, scores


# ---------------------------------------------------------------- full decoder
class CaptionModel(keras.Model):
    """(image_features, words_so_far) -> scores for the next word at every step."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.image_encoder = ImageEncoder(d_model, num_heads, ff_dim, dropout)
        self.embedding = TokenAndPositionEmbedding(vocab_size, max_len, d_model)
        self.embed_drop = layers.Dropout(dropout)
        self.blocks = [
            DecoderBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ]
        self.out = layers.Dense(vocab_size)

    def call(self, inputs, training=False, return_attention=False):
        features, tokens = inputs
        image = self.image_encoder(features, training=training)
        x = self.embed_drop(self.embedding(tokens), training=training)
        scores = None
        for block in self.blocks:
            x, scores = block(x, image, training=training)
        logits = self.out(x)
        return (logits, scores) if return_attention else logits


def build_caption_model(config: dict) -> CaptionModel:
    """Create the decoder and build its weights with a dummy forward pass."""
    model = CaptionModel(
        vocab_size=config["vocab_size"],
        max_len=config["max_len"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    dummy_feats = np.zeros((1, config["num_regions"], config["feature_dim"]), "float32")
    model((dummy_feats, np.zeros((1, 1), "int32")))
    return model


# ---------------------------------------------------------------- loss / metric
def masked_loss(y_true, y_pred):
    """Cross-entropy that ignores <pad> positions (id 0)."""
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = ops.cast(ops.not_equal(y_true, 0), loss.dtype)
    return ops.sum(loss * mask) / ops.maximum(ops.sum(mask), 1.0)


def masked_accuracy(y_true, y_pred):
    """Next-word accuracy, ignoring <pad> positions."""
    y_true = ops.cast(y_true, "int32")
    pred = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
    mask = ops.cast(ops.not_equal(y_true, 0), "float32")
    hits = ops.cast(ops.equal(y_true, pred), "float32") * mask
    return ops.sum(hits) / ops.maximum(ops.sum(mask), 1.0)
