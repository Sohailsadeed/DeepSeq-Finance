"""Global configuration for financial forecasting models and data pipelines.
Defines hyperparameters, window sizes, split ratios, and feature sets."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "btc_1d_data_2018_to_2025.csv"
PROCESSED_PATH = BASE_DIR / "processed_data.npz"
RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SPLIT_RATIOS = (0.70, 0.15, 0.15)
WINDOW = 30
SEED = 42

TASKS = ("binary", "direction3", "regression")
FLAT_BAND = 0.005

REQUIRED_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
OPTIONAL_FEATURES = ["Quote asset volume", "Number of trades"]
BASE_FEATURES = REQUIRED_FEATURES + OPTIONAL_FEATURES

DERIVED_FEATURES = [
    "Daily return",
    "High-low pct range",
    "MA 7",
    "MA 30",
    "Rolling vol 7",
    "Volume change",
    "Overnight return",
    "RSI 14",
    "Bollinger pct b",
    "Garman-Klass vol",
    "Sin hour",
    "Cos hour",
    "Sin dayofweek",
    "Cos dayofweek",
    "Is weekend",
]

FEATURES = BASE_FEATURES + DERIVED_FEATURES
TARGET = "Target"

HIDDEN_SIZE = 48
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 40
PATIENCE = 5
MIN_DELTA = 1e-4

BINARY_LABELS = ("DOWN", "UP")
DIRECTION3_LABELS = ("DOWN", "FLAT", "UP")
TASK_LOSS = {"binary": "bce", "direction3": "ce", "regression": "mse"}

SUGGEST_P_THRESHOLD = 0.45
SUGGEST_RET_THRESHOLD = 0.0
