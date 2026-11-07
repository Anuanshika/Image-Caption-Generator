"""Evaluate captions with BLEU scores and save example pictures.

Usage:
    python evaluate.py                    # downloads Flickr8k from Kaggle
    python evaluate.py --data-dir PATH    # use a local copy
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu  # noqa: E402
from PIL import Image  # noqa: E402

from captioner import Captioner  # noqa: E402
from config import REPORTS_DIR  # noqa: E402
from tokenizer import clean_caption  # noqa: E402

BLEU_WEIGHTS = {
    "BLEU-1": (1, 0, 0, 0),
    "BLEU-2": (0.5, 0.5, 0, 0),
    "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0),
    "BLEU-4": (0.25, 0.25, 0.25, 0.25),
}


def bleu_scores(references: list[list[list[str]]], hypotheses: list[list[str]]) -> dict:
    smooth = SmoothingFunction().method1
    return {k: float(corpus_bleu(references, hypotheses, weights=w, smoothing_function=smooth))
            for k, w in BLEU_WEIGHTS.items()}


def evaluate_captioner(
    captioner: Captioner,
    test_features: np.ndarray,
    test_names: list[str],
    captions: dict[str, list[str]],
    image_dir: Path,
    out_dir: Path = REPORTS_DIR,
    beam_samples: int = 200,
    beam_width: int = 3,
) -> dict:
    """BLEU for greedy (all test images) and beam search (first `beam_samples`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = [[clean_caption(c) for c in captions[n]] for n in test_names]

    greedy_ids: list[list[int]] = []
    for i in range(0, len(test_names), 64):
        feats = test_features[i: i + 64].astype("float32")
        greedy_ids.extend(captioner.greedy_batch(feats))
    greedy = [captioner.vocab.decode(ids) for ids in greedy_ids]

    n_beam = min(beam_samples, len(test_names))
    beam = []
    for i in range(n_beam):
        feats = test_features[i: i + 1].astype("float32")
        beam.append(captioner.vocab.decode(captioner.beam_search(feats, beam_width)[0][0]))

    results = {
        "greedy": bleu_scores(refs, greedy),
        f"beam{beam_width}_first{n_beam}": bleu_scores(refs[:n_beam], beam),
        "greedy_first_same_images": bleu_scores(refs[:n_beam], greedy[:n_beam]),
        "num_test_images": len(test_names),
    }
    print(json.dumps(results, indent=2))
    (out_dir / "bleu_scores.json").write_text(json.dumps(results, indent=2))

    # Example grid: generated (beam) vs one human caption.
    n_show = min(6, n_beam)
    if n_show:
        cols = 3
        rows = int(np.ceil(n_show / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        for ax, i in zip(np.atleast_1d(axes).ravel(), range(rows * cols)):
            ax.axis("off")
            if i >= n_show:
                continue
            ax.imshow(Image.open(image_dir / test_names[i]).convert("RGB"))
            text = (f"Model: {' '.join(beam[i])}\n"
                    f"Human: {captions[test_names[i]][0]}")
            ax.set_title(textwrap.fill(text, 45), fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(out_dir / "examples.png", dpi=120)
        plt.close(fig)
    print(f"Reports saved to {out_dir}/")
    return results


def main() -> None:
    from config import CACHE_DIR
    from data_utils import download_dataset, load_captions, split_images
    from features import cached_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--beam-samples", type=int, default=200)
    args = parser.parse_args()

    captioner = Captioner()
    image_dir, captions = load_captions(args.data_dir or download_dataset())
    _, _, test_names = split_images(list(captions))
    all_names = sorted(captions)
    all_feats = cached_features(
        captioner.encoder, [image_dir / n for n in all_names],
        captioner.config["img_size"], CACHE_DIR / "features_all.npy",
    )
    row = {n: i for i, n in enumerate(all_names)}
    test_feats = all_feats[[row[n] for n in test_names]]
    evaluate_captioner(captioner, test_feats, test_names, captions, image_dir,
                       beam_samples=args.beam_samples)


if __name__ == "__main__":
    main()
