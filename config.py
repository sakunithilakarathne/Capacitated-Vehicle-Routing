import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATASET = os.path.join(RAW_DATA_DIR, "A-n32-k5.vrp")

DIST_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "dist_matrix.npy")
DEMANDS_PATH = os.path.join(ARTIFACTS_DIR, "demands.npy")
COORDS_PATH = os.path.join(ARTIFACTS_DIR, "coords.npy")