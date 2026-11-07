"""Shared paths and default settings."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
REPORTS_DIR = ROOT_DIR / "reports"
CACHE_DIR = ROOT_DIR / "cache"  # pre-extracted image features (git-ignored)

ENCODER_WEIGHTS = "encoder.weights.h5"
DECODER_WEIGHTS = "decoder.weights.h5"
CONFIG_FILE = "config.json"
VOCAB_FILE = "vocab.json"

# Public Kaggle dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k
KAGGLE_DATASET = "adityajn105/flickr8k"

IMG_SIZE = 224
MAX_LEN = 30          # tokens per caption incl. <start> and <end>
MIN_WORD_FREQ = 5     # rarer words become <unk>
SEED = 42

# Transformer decoder size
D_MODEL = 256
NUM_HEADS = 4
FF_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2
