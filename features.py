"""Run every image through the CNN once and cache the region features."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


def _load_image(path, img_size: int):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.image.resize(tf.cast(img, tf.float32), (img_size, img_size))


def extract_features(
    encoder: keras.Model, image_paths: list[Path], img_size: int, batch_size: int = 64
) -> np.ndarray:
    """Return float16 features of shape (n_images, regions, channels)."""
    ds = (
        tf.data.Dataset.from_tensor_slices([str(p) for p in image_paths])
        .map(lambda p: _load_image(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    print(f"Extracting CNN features for {len(image_paths)} images...")
    return encoder.predict(ds, verbose=1).astype(np.float16)


def cached_features(
    encoder: keras.Model, image_paths: list[Path], img_size: int, cache_file: Path
) -> np.ndarray:
    """Load features from `cache_file` if it matches, otherwise extract and save."""
    names_file = cache_file.with_suffix(".names.txt")
    names = [p.name for p in image_paths]
    if cache_file.exists() and names_file.exists():
        if names_file.read_text().splitlines() == names:
            print(f"Using cached features: {cache_file}")
            return np.load(cache_file)
    feats = extract_features(encoder, image_paths, img_size)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, feats)
    names_file.write_text("\n".join(names))
    return feats
