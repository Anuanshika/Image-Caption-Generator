"""Train the image caption generator on Flickr8k.

Usage:
    python train.py                       # download from Kaggle and train
    python train.py --epochs 30           # train longer
    python train.py --data-dir PATH       # use an already downloaded dataset

Outputs (model/ folder, used by the Streamlit app):
    encoder.weights.h5   CNN image encoder (EfficientNetB0)
    decoder.weights.h5   Transformer caption decoder
    config.json          model settings
    vocab.json           word list
Reports (reports/ folder): training curves, BLEU scores, example captions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import config as C  # noqa: E402
from captioner import Captioner  # noqa: E402
from data_utils import CaptionBatches, download_dataset, load_captions, split_images  # noqa: E402
from evaluate import evaluate_captioner  # noqa: E402
from features import cached_features  # noqa: E402
from model import build_caption_model, build_cnn_encoder, masked_accuracy, masked_loss  # noqa: E402
from tokenizer import Vocabulary  # noqa: E402


def encode_split(names, captions, vocab, row, max_len):
    """One training example per (image, caption) pair."""
    image_index, sequences = [], []
    for n in names:
        for cap in captions[n]:
            image_index.append(row[n])
            sequences.append(vocab.encode(cap, max_len))
    return np.array(image_index), np.array(sequences, dtype="int32")


def plot_history(hist: dict, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in ((ax1, "loss", "Loss"), (ax2, "masked_accuracy", "Next-word accuracy")):
        ax.plot(hist[key], label="train")
        ax.plot(hist[f"val_{key}"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the image caption generator")
    p.add_argument("--data-dir", type=Path, default=None, help="local dataset folder")
    p.add_argument("--kaggle-dataset", default=C.KAGGLE_DATASET)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-word-freq", type=int, default=C.MIN_WORD_FREQ)
    p.add_argument("--beam-samples", type=int, default=200,
                   help="test images scored with beam search")
    p.add_argument("--no-pretrained", action="store_true",
                   help="random CNN weights (only for quick offline tests)")
    args = p.parse_args()

    keras.utils.set_random_seed(C.SEED)
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | "
          f"GPUs: {tf.config.list_physical_devices('GPU')}")

    # 1. Data --------------------------------------------------------------
    image_dir, captions = load_captions(args.data_dir or download_dataset(args.kaggle_dataset))
    train_names, val_names, test_names = split_images(list(captions))
    print(f"Images -> train {len(train_names)}, val {len(val_names)}, test {len(test_names)}")

    vocab = Vocabulary.build((c for n in train_names for c in captions[n]), args.min_word_freq)
    print(f"Vocabulary: {len(vocab)} words (min frequency {args.min_word_freq})")

    # 2. CNN features (computed once, cached) ------------------------------
    encoder = build_cnn_encoder(C.IMG_SIZE, pretrained=not args.no_pretrained)
    all_names = sorted(captions)
    cache = C.CACHE_DIR / ("features_all.npy" if not args.no_pretrained else "features_all_random.npy")
    features = cached_features(encoder, [image_dir / n for n in all_names], C.IMG_SIZE, cache)
    row = {n: i for i, n in enumerate(all_names)}

    train_idx, train_seq = encode_split(train_names, captions, vocab, row, C.MAX_LEN)
    val_idx, val_seq = encode_split(val_names, captions, vocab, row, C.MAX_LEN)
    train_data = CaptionBatches(features, train_idx, train_seq, args.batch_size, shuffle=True)
    val_data = CaptionBatches(features, val_idx, val_seq, args.batch_size, shuffle=False)

    # 3. Model -------------------------------------------------------------
    config = {
        "img_size": C.IMG_SIZE, "max_len": C.MAX_LEN, "vocab_size": len(vocab),
        "num_regions": int(features.shape[1]), "feature_dim": int(features.shape[2]),
        "d_model": C.D_MODEL, "num_heads": C.NUM_HEADS, "ff_dim": C.FF_DIM,
        "num_layers": C.NUM_LAYERS, "dropout": C.DROPOUT,
        "trained_with": {"tensorflow": tf.__version__, "keras": keras.__version__},
    }
    model = build_caption_model(config)
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=1.0),
                  loss=masked_loss, metrics=[masked_accuracy])
    model.summary()

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    decoder_path = C.MODEL_DIR / C.DECODER_WEIGHTS
    callbacks = [
        keras.callbacks.ModelCheckpoint(decoder_path, monitor="val_loss",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                          min_lr=1e-6, verbose=1),
    ]

    # 4. Train -------------------------------------------------------------
    start = time.time()
    history = model.fit(train_data, validation_data=val_data, epochs=args.epochs,
                        callbacks=callbacks)
    print(f"Training took {(time.time() - start) / 60:.1f} min")

    # 5. Save everything the app needs ------------------------------------
    # Save weights from an uncompiled copy so optimizer state is not stored (smaller file).
    clean = build_caption_model(config)
    clean.set_weights(model.get_weights())
    clean.save_weights(decoder_path)
    encoder.save_weights(C.MODEL_DIR / C.ENCODER_WEIGHTS)
    (C.MODEL_DIR / C.CONFIG_FILE).write_text(json.dumps(config, indent=2))
    vocab.save(C.MODEL_DIR / C.VOCAB_FILE)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (C.REPORTS_DIR / "history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, C.REPORTS_DIR / "training_curves.png")
    print(f"Model saved to {C.MODEL_DIR}/")

    # 6. Evaluate on unseen test images -----------------------------------
    captioner = Captioner(encoder=encoder, decoder=model, config=config, vocab=vocab)
    test_feats = features[[row[n] for n in test_names]]
    evaluate_captioner(captioner, test_feats, test_names, captions, image_dir,
                       beam_samples=args.beam_samples)


if __name__ == "__main__":
    main()
