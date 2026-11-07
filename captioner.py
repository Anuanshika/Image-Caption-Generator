"""Inference: load the trained model and caption an image (greedy or beam search)."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from config import CONFIG_FILE, DECODER_WEIGHTS, ENCODER_WEIGHTS, MODEL_DIR, VOCAB_FILE
from model import build_caption_model, build_cnn_encoder
from tokenizer import Vocabulary

# The first decoding step attends over a single token; Keras warns about it harmlessly.
warnings.filterwarnings("ignore", message="You are using a softmax over axis")


def model_files_exist(model_dir: Path = MODEL_DIR) -> bool:
    return all((Path(model_dir) / f).exists()
               for f in (ENCODER_WEIGHTS, DECODER_WEIGHTS, CONFIG_FILE, VOCAB_FILE))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


@dataclass
class CaptionResult:
    caption: str
    words: list[str]
    token_ids: list[int]
    alternatives: list[tuple[str, float]] = field(default_factory=list)


class Captioner:
    """Wraps the CNN encoder + Transformer decoder for easy captioning."""

    def __init__(self, model_dir: Path = MODEL_DIR, encoder=None, decoder=None,
                 config: dict | None = None, vocab: Vocabulary | None = None) -> None:
        model_dir = Path(model_dir)
        self.config = config or json.loads((model_dir / CONFIG_FILE).read_text())
        self.vocab = vocab or Vocabulary.load(model_dir / VOCAB_FILE)
        if encoder is None:
            encoder = build_cnn_encoder(self.config["img_size"], pretrained=False)
            encoder.load_weights(model_dir / ENCODER_WEIGHTS)
        if decoder is None:
            decoder = build_caption_model(self.config)
            decoder.load_weights(model_dir / DECODER_WEIGHTS)
        self.encoder, self.decoder = encoder, decoder
        self.max_len = self.config["max_len"]
        # Never generate these tokens.
        self._banned = [self.vocab.pad_id, self.vocab.start_id, self.vocab.unk_id]

    # ------------------------------------------------------------ image -> features
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Same resize as training (tf.image bilinear), pixels kept 0-255."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        arr = tf.image.resize(np.asarray(image, dtype=np.float32),
                              (self.config["img_size"], self.config["img_size"]))
        return arr.numpy()[np.newaxis]

    def encode_image(self, image: Image.Image) -> np.ndarray:
        return self.encoder(self.preprocess(image), training=False).numpy()

    # ------------------------------------------------------------ decoding
    def _next_logprobs(self, feats: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        logits = self.decoder((feats, tokens), training=False)[:, -1, :].numpy()
        logp = _log_softmax(logits.astype(np.float64))
        logp[:, self._banned] = -np.inf
        return logp

    def greedy_batch(self, feats: np.ndarray) -> list[list[int]]:
        """Greedy decoding for a batch of images (pick the best word each step)."""
        n = feats.shape[0]
        tokens = np.full((n, 1), self.vocab.start_id, dtype="int32")
        done = np.zeros(n, dtype=bool)
        for _ in range(self.max_len - 1):
            nxt = self._next_logprobs(feats, tokens).argmax(axis=-1).astype("int32")
            nxt[done] = self.vocab.pad_id
            tokens = np.concatenate([tokens, nxt[:, None]], axis=1)
            done |= nxt == self.vocab.end_id
            if done.all():
                break
        return [self._strip(t[1:]) for t in tokens]

    def beam_search(self, feats: np.ndarray, beam_width: int = 3,
                    alpha: float = 0.7) -> list[tuple[list[int], float]]:
        """Keep the `beam_width` best partial captions at every step.

        Returns (token_ids, score) pairs, best first. Scores are length-normalised
        log-probabilities, so longer captions are not unfairly penalised.
        """
        beams: list[tuple[list[int], float]] = [([self.vocab.start_id], 0.0)]
        finished: list[tuple[list[int], float]] = []
        for _ in range(self.max_len - 1):
            seqs = np.array([b[0] for b in beams], dtype="int32")
            logp = self._next_logprobs(np.repeat(feats, len(beams), axis=0), seqs)
            candidates = []
            for i, (seq, score) in enumerate(beams):
                for tok in np.argsort(logp[i])[-beam_width:]:
                    candidates.append((seq + [int(tok)], score + float(logp[i, tok])))
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in candidates:
                (finished if seq[-1] == self.vocab.end_id else beams).append((seq, score))
                if len(beams) == beam_width:
                    break
            if not beams or len(finished) >= beam_width:
                break
        pool = finished or beams
        ranked = [(self._strip(s[1:]), sc / max(len(s) - 1, 1) ** alpha) for s, sc in pool]
        ranked.sort(key=lambda r: r[1], reverse=True)
        return ranked

    def _strip(self, ids) -> list[int]:
        out = []
        for i in ids:
            if int(i) in (self.vocab.end_id, self.vocab.pad_id):
                break
            out.append(int(i))
        return out

    # ------------------------------------------------------------ public API
    def caption(self, image: Image.Image, method: str = "beam",
                beam_width: int = 3) -> tuple[CaptionResult, np.ndarray]:
        """Caption one PIL image. Returns (result, image_features)."""
        feats = self.encode_image(image)
        return self.caption_features(feats, method, beam_width), feats

    def caption_features(self, feats: np.ndarray, method: str = "beam",
                         beam_width: int = 3) -> CaptionResult:
        if method == "greedy":
            ids = self.greedy_batch(feats)[0]
            alternatives: list[tuple[str, float]] = []
        else:
            ranked = self.beam_search(feats, beam_width)
            ids = ranked[0][0]
            alternatives = [(" ".join(self.vocab.decode(s)), float(sc)) for s, sc in ranked]
        words = self.vocab.decode(ids)
        text = " ".join(words)
        return CaptionResult(text[:1].upper() + text[1:], words, ids, alternatives)

    def attention_maps(self, feats: np.ndarray, token_ids: list[int]) -> np.ndarray:
        """Where the model looked for each word: array (n_words, grid, grid), 0-1."""
        if not token_ids:
            return np.zeros((0, 1, 1))
        tokens = np.array([[self.vocab.start_id] + token_ids], dtype="int32")
        _, scores = self.decoder((feats, tokens), training=False, return_attention=True)
        maps = scores.numpy()[0].mean(axis=0)  # average heads -> (steps, regions)
        maps = maps[: len(token_ids)]  # step i is where it looked to write word i
        grid = int(round(np.sqrt(maps.shape[-1])))
        maps = maps.reshape(len(token_ids), grid, grid)
        maps = maps - maps.min(axis=(1, 2), keepdims=True)
        return maps / np.maximum(maps.max(axis=(1, 2), keepdims=True), 1e-8)
