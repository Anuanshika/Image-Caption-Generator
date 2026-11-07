"""Download Flickr8k from Kaggle, read captions, split, and batch the data."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import keras
import numpy as np

from config import KAGGLE_DATASET, SEED

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def download_dataset(slug: str = KAGGLE_DATASET) -> Path:
    """Download a public Kaggle dataset with kagglehub (cached after first run)."""
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("kagglehub missing. Run: pip install kagglehub") from exc
    print(f"Downloading Kaggle dataset '{slug}'...")
    try:
        path = Path(kagglehub.dataset_download(slug))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Kaggle download failed: {exc}\nFix: run `import kagglehub; "
            "kagglehub.login()` once, or set KAGGLE_USERNAME / KAGGLE_KEY."
        ) from exc
    print(f"Dataset ready at: {path}")
    return path


def _find_image_dir(root: Path) -> Path:
    best, best_count = None, 0
    for d in [root, *root.rglob("*")]:
        if d.is_dir():
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
            if n > best_count:
                best, best_count = d, n
    if best is None:
        raise FileNotFoundError(f"No images found under {root}")
    return best


def load_captions(root: Path) -> tuple[Path, dict[str, list[str]]]:
    """Return (image_folder, {image_file_name: [caption, ...]}).

    Supports both Kaggle formats: `captions.txt` (CSV: image,caption) and the
    original `Flickr8k.token.txt` (image.jpg#0<TAB>caption).
    """
    root = Path(root)
    image_dir = _find_image_dir(root)
    captions: dict[str, list[str]] = {}

    csv_files = sorted(root.rglob("captions.txt"))
    token_files = sorted(root.rglob("Flickr8k.token.txt"))
    if csv_files:
        with csv_files[0].open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                captions.setdefault(row["image"].strip(), []).append(row["caption"].strip())
    elif token_files:
        for line in token_files[0].read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                key, cap = line.split("\t", 1)
                captions.setdefault(key.split("#")[0].strip(), []).append(cap.strip())
    else:
        raise FileNotFoundError(f"No captions.txt or Flickr8k.token.txt under {root}")

    existing = {f.name for f in image_dir.iterdir()}
    captions = {k: v for k, v in captions.items() if k in existing and v}
    print(f"Found {len(captions)} images with captions in {image_dir}")
    return image_dir, captions


def split_images(
    names: list[str], val_frac: float = 0.1, test_frac: float = 0.1
) -> tuple[list[str], list[str], list[str]]:
    """Split image names (not captions) so no image leaks between splits."""
    names = sorted(names)
    rng = np.random.default_rng(SEED)
    rng.shuffle(names)
    n_test = max(1, int(len(names) * test_frac))
    n_val = max(1, int(len(names) * val_frac))
    return names[n_test + n_val:], names[n_test: n_test + n_val], names[:n_test]


class CaptionBatches(keras.utils.PyDataset):
    """Yields ((image_features, caption_in), caption_out) batches.

    caption_in  = <start> w1 w2 ... (model input)
    caption_out = w1 w2 ... <end>   (what the model must predict next)
    """

    def __init__(
        self,
        features: np.ndarray,
        image_index: np.ndarray,
        sequences: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.features, self.image_index, self.sequences = features, image_index, sequences
        self.batch_size, self.shuffle = batch_size, shuffle
        self.order = np.arange(len(sequences))
        self.rng = np.random.default_rng(SEED)
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)

    def __getitem__(self, i: int):
        idx = self.order[i * self.batch_size: (i + 1) * self.batch_size]
        feats = self.features[self.image_index[idx]].astype("float32")
        seqs = self.sequences[idx]
        return (feats, seqs[:, :-1]), seqs[:, 1:]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)
