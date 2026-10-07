"""
Configuration constants for the Entity Resolution pipeline.
All hyperparameters and paths in one place.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_RESOURCE = os.path.join(BASE_DIR, "student_resource")
TRAIN_DIR = os.path.join(STUDENT_RESOURCE, "dataset", "train")
TEST_DIR = os.path.join(STUDENT_RESOURCE, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# ─── Blocking Hyperparameters ────────────────────────────────────────────────
TFIDF_TOP_K = 20                # Top-K candidates from TF-IDF blocking
TFIDF_NGRAM_RANGE = (2, 4)      # Character n-gram range for TF-IDF
TFIDF_MAX_FEATURES = 80000      # Max vocabulary size for TF-IDF
BLOCKING_CHUNK_SIZE = 5000      # Process S1 entities in chunks during blocking

# ─── Classifier Hyperparameters ──────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': 5,
    'verbose': -1,
    'n_jobs': -1,
    'num_iterations': 500,
    'early_stopping_rounds': 30,
}

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5,
    'n_estimators': 500,
    'early_stopping_rounds': 30,
    'n_jobs': -1,
    'verbosity': 0,
}

# ─── Threshold Defaults ─────────────────────────────────────────────────────
DEFAULT_MATCH_THRESHOLD = 0.5   # Will be optimized during training
THRESHOLD_SEARCH_RANGE = (0.30, 0.95)
THRESHOLD_SEARCH_STEP = 0.01

# ─── Validation ──────────────────────────────────────────────────────────────
VALIDATION_FRACTION = 0.15      # 15% of training S1 entities for validation
RANDOM_SEED = 42

# ─── Processing ──────────────────────────────────────────────────────────────
FEATURE_CHUNK_SIZE = 50000      # Pairs to featurize in one batch
MAX_CANDIDATES_PER_S1 = 50      # Safety cap on candidates per S1 entity
