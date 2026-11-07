"""Fast tests that need no dataset and no trained model.  Run: pytest -q"""

import numpy as np
import keras
from PIL import Image

from captioner import Captioner
from data_utils import CaptionBatches, split_images
from model import build_caption_model, build_cnn_encoder, masked_accuracy, masked_loss
from tokenizer import Vocabulary, clean_caption
from viz import attention_overlay

COLORS = ["red", "green", "blue", "yellow"]


def test_clean_caption():
    assert clean_caption("A dog, running!! in 2 parks .") == ["a", "dog", "running", "in", "parks"]


def test_vocab_roundtrip(tmp_path):
    vocab = Vocabulary.build(["a dog runs", "a dog sits", "a cat"], min_freq=2)
    assert {"a", "dog"} <= set(vocab.itos) and "cat" not in vocab.itos
    ids = vocab.encode("A dog jumps", max_len=8)
    assert len(ids) == 8 and ids[0] == vocab.start_id
    assert vocab.decode(ids) == ["a", "dog", "<unk>"]
    vocab.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").itos == vocab.itos


def test_split_has_no_overlap():
    names = [f"{i}.jpg" for i in range(100)]
    tr, va, te = split_images(names)
    assert len(tr) + len(va) + len(te) == 100
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))


def _toy_problem():
    """Image features that encode a colour; captions that name it."""
    caps = {c: [f"a plain {c} background", f"a bright {c} square"] for c in COLORS}
    vocab = Vocabulary.build([x for v in caps.values() for x in v], 1)
    rng = np.random.default_rng(0)
    feats = np.zeros((40, 49, 32), "float16")
    idx, seq = [], []
    for i in range(40):
        feats[i, :, (i % 4) * 8:(i % 4 + 1) * 8] = 1
        feats[i] += rng.normal(0, 0.1, (49, 32)).astype("float16")
        for c in caps[COLORS[i % 4]]:
            idx.append(i)
            seq.append(vocab.encode(c, 12))
    cfg = dict(img_size=224, max_len=12, vocab_size=len(vocab), num_regions=49,
               feature_dim=32, d_model=64, num_heads=2, ff_dim=128, num_layers=2, dropout=0.1)
    return feats, np.array(idx), np.array(seq), vocab, cfg


def test_decoder_learns_to_describe_the_image():
    keras.utils.set_random_seed(0)
    feats, idx, seq, vocab, cfg = _toy_problem()
    model = build_caption_model(cfg)
    model.compile(keras.optimizers.Adam(2e-3), loss=masked_loss, metrics=[masked_accuracy])
    model.fit(CaptionBatches(feats, idx, seq, 16), epochs=25, verbose=0)

    cap = Captioner(encoder=object(), decoder=model, config=cfg, vocab=vocab)
    for i, colour in enumerate(COLORS):
        f = feats[i:i + 1].astype("float32")
        beam = cap.caption_features(f, "beam", 3)
        greedy = cap.caption_features(f, "greedy")
        assert colour in beam.words and colour in greedy.words
        maps = cap.attention_maps(f, beam.token_ids)
        assert maps.shape == (len(beam.token_ids), 7, 7)
        assert np.all((maps >= 0) & (maps <= 1))


def test_save_load_and_caption_real_image(tmp_path):
    """Full pipeline from saved files, like the app does (random CNN weights)."""
    import json

    _, _, _, vocab, cfg = _toy_problem()
    cfg.update(feature_dim=1280)
    encoder = build_cnn_encoder(224, pretrained=False)
    decoder = build_caption_model(cfg)
    encoder.save_weights(tmp_path / "encoder.weights.h5")
    decoder.save_weights(tmp_path / "decoder.weights.h5")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    vocab.save(tmp_path / "vocab.json")

    cap = Captioner(tmp_path)
    img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    result, feats = cap.caption(img, "beam", 2)
    assert feats.shape == (1, 49, 1280)
    assert isinstance(result.caption, str) and len(result.token_ids) <= cfg["max_len"]
    overlay = attention_overlay(img, np.random.rand(7, 7))
    assert overlay.mode == "RGB" and max(overlay.size) <= 320
