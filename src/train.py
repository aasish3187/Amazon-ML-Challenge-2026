"""
Train LightGBM Models for Business Entity Resolution.
Trains separate models for US and India, with early stopping,
threshold search optimizing macro F₀.₅, and singleton handling.
"""
import os
import sys
import time
import pickle
import random
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.blocking import run_full_blocking
from src.evaluate import compute_macro_f05, evaluate_from_dicts

sys.stdout.reconfigure(encoding='utf-8')


def build_country_training_data(country: str, n_train_s1: int = 5000, n_val_s1: int = 1500,
                                n_distractors: int = 25000, seed: int = 42):
    """
    Sample S1 entities for a country, gather their true matches from S2/S3,
    add distractors, and split into train/val.
    """
    print(f"\n{'='*70}")
    print(f"  BUILDING DATASET FOR {country} (Train S1: {n_train_s1:,}, Val S1: {n_val_s1:,})")
    print(f"{'='*70}")
    random.seed(seed)
    t0 = time.time()

    target_s1_count = n_train_s1 + n_val_s1
    s1_recs = []

    # Stream S1 for this country
    with open(TRAIN_S1, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 3 and parts[3].strip() == country:
                s1_recs.append(normalize_record(parts[0], parts[1],
                                                parts[2] if len(parts) > 2 else '',
                                                country))
                if len(s1_recs) >= target_s1_count:
                    break

    print(f"  Loaded {len(s1_recs):,} S1 entities for {country}")
    s1_ids = {r['entity_id']: r for r in s1_recs}

    # Load Ground Truth for these S1 records
    target_gt = {}
    needed_matches = set()
    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0]
            if s1_id in s1_ids:
                matches = parts[1].split(',') if len(parts) > 1 and parts[1] else []
                target_gt[s1_id] = set(matches)
                for m in matches:
                    needed_matches.add(m)

    n_matched_s1 = sum(1 for m in target_gt.values() if len(m) > 0)
    print(f"  True matched S1: {n_matched_s1:,}, Singletons: {len(s1_recs)-n_matched_s1:,}")
    print(f"  Total true match candidates needed from S2/S3: {len(needed_matches):,}")

    # Scan S2 and S3 for true matches + random distractors
    pool_matches = []
    distractors = []

    for path in [TRAIN_S2, TRAIN_S3]:
        print(f"  Scanning {os.path.basename(path)}...")
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                c = parts[3].strip() if len(parts) > 3 else ''
                if c != country:
                    continue
                eid = parts[0]
                if eid in needed_matches:
                    pool_matches.append(normalize_record(eid, parts[1],
                                                         parts[2] if len(parts) > 2 else '',
                                                         country))
                elif len(distractors) < n_distractors and random.random() < 0.02:
                    distractors.append(normalize_record(eid, parts[1],
                                                        parts[2] if len(parts) > 2 else '',
                                                        country))

    full_pool = pool_matches + distractors
    random.shuffle(full_pool)
    print(f"  Total pool: {len(full_pool):,} ({len(pool_matches):,} true matches + {len(distractors):,} distractors)")
    print(f"  Pool gathered in {time.time()-t0:.1f}s")

    # Split S1 into Train and Val
    random.seed(seed)
    shuffled_s1 = list(s1_recs)
    random.shuffle(shuffled_s1)

    train_s1 = shuffled_s1[:n_train_s1]
    val_s1 = shuffled_s1[n_train_s1:n_train_s1 + n_val_s1]

    train_gt = {r['entity_id']: target_gt.get(r['entity_id'], set()) for r in train_s1}
    val_gt = {r['entity_id']: target_gt.get(r['entity_id'], set()) for r in val_s1}

    return train_s1, val_s1, full_pool, train_gt, val_gt


def featurize_candidates(s1_records, pool_records, blocking_results, gt_map=None):
    """
    Featurize candidate pairs from blocking.
    Returns:
        X: np.ndarray of shape (N, 18)
        y: np.ndarray of shape (N,) if gt_map else None
        pair_ids: list of (s1_id, pool_id)
    """
    X_rows = []
    y_rows = []
    pair_ids = []

    for s1_idx, pool_indices in blocking_results.items():
        s1_rec = s1_records[s1_idx]
        s1_id = s1_rec['entity_id']
        true_set = gt_map.get(s1_id, set()) if gt_map is not None else None

        for pool_idx in pool_indices:
            pool_rec = pool_records[pool_idx]
            pool_id = pool_rec['entity_id']

            feats = compute_features(s1_rec, pool_rec)
            feat_vals = [feats[fn] for fn in FEATURE_NAMES]

            X_rows.append(feat_vals)
            pair_ids.append((s1_id, pool_id))

            if true_set is not None:
                y_rows.append(1 if pool_id in true_set else 0)

    X = np.array(X_rows, dtype=np.float32) if X_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32) if y_rows else None

    return X, y, pair_ids


def train_and_optimize(country: str, train_s1, val_s1, pool, train_gt, val_gt):
    """
    Run blocking, featurize, train LightGBM, and find optimal threshold for country.
    """
    import lightgbm as lgb

    print(f"\n--- [1] Blocking Train & Val S1 ---")
    train_cands = run_full_blocking(train_s1, pool, top_k=25, max_candidates=75)
    val_cands = run_full_blocking(val_s1, pool, top_k=25, max_candidates=75)

    print(f"\n--- [2] Featurizing Pairs ---")
    t0 = time.time()
    X_train, y_train, train_pairs = featurize_candidates(train_s1, pool, train_cands, train_gt)
    X_val, y_val, val_pairs = featurize_candidates(val_s1, pool, val_cands, val_gt)

    pos_tr = int(y_train.sum()) if y_train is not None else 0
    pos_val = int(y_val.sum()) if y_val is not None else 0
    print(f"  Train pairs: {len(X_train):,} ({pos_tr:,} pos, {len(X_train)-pos_tr:,} neg)")
    print(f"  Val pairs  : {len(X_val):,} ({pos_val:,} pos, {len(X_val)-pos_val:,} neg)")
    print(f"  Featurized in {time.time()-t0:.1f}s")

    print(f"\n--- [3] Training LightGBM Classifier ---")
    train_data = lgb.Dataset(X_train, y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, y_val, feature_name=FEATURE_NAMES, reference=train_data)

    params = LIGHTGBM_PARAMS.copy()
    n_iter = params.pop('num_iterations', 600)
    early = params.pop('early_stopping_rounds', 40)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_iter,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    print(f"\n  Top 10 Features by Gain ({country}):")
    for fn, g in feat_imp[:10]:
        print(f"    {fn:30s}: {g:12.1f}")

    print(f"\n--- [4] Optimizing Threshold for Macro F₀.₅ ---")
    val_preds_proba = model.predict(X_val)

    s1_scores = defaultdict(list)
    for (s1_id, pool_id), score in zip(val_pairs, val_preds_proba):
        s1_scores[s1_id].append((score, pool_id))

    all_val_s1_ids = [r['entity_id'] for r in val_s1]

    best_thresh = 0.50
    best_f05 = -1.0
    best_summary = None

    for thresh in np.arange(0.20, 0.90, 0.02):
        predictions = {}
        for s1_id in all_val_s1_ids:
            cands = s1_scores.get(s1_id, [])
            matched = {cand_id for sc, cand_id in cands if sc >= thresh}
            predictions[s1_id] = matched

        f05 = compute_macro_f05(predictions, val_gt)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(thresh)

    print(f"  >>> Optimal Threshold for {country}: {best_thresh:.2f} <<<")
    print(f"  >>> Validation Macro F₀.₅: {best_f05:.4f} <<<")

    # Evaluate best threshold with breakdown
    best_predictions = {}
    for s1_id in all_val_s1_ids:
        cands = s1_scores.get(s1_id, [])
        best_predictions[s1_id] = {cand_id for sc, cand_id in cands if sc >= best_thresh}

    print(f"\n  Final Validation Report for {country}:")
    evaluate_from_dicts(best_predictions, val_gt, verbose=True)

    return model, best_thresh, best_f05


def main():
    total_start = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — CHAMPIONSHIP MODEL TRAINING")
    print("=" * 70)

    models = {}
    thresholds = {}
    f05_scores = {}

    for country in ['US', 'India']:
        train_s1, val_s1, pool, train_gt, val_gt = build_country_training_data(
            country,
            n_train_s1=6000,
            n_val_s1=2000,
            n_distractors=25000,
            seed=42,
        )

        model, thresh, f05 = train_and_optimize(
            country, train_s1, val_s1, pool, train_gt, val_gt
        )

        models[country] = model
        thresholds[country] = thresh
        f05_scores[country] = f05

    # Save to models.pkl
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'models.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'models': models,
            'thresholds': thresholds,
            'f05_scores': f05_scores,
            'features': FEATURE_NAMES,
        }, f)

    print(f"\n{'='*70}")
    print(f"  🏆 TRAINING COMPLETE! Models saved to {model_path}")
    for c in ['US', 'India']:
        print(f"    {c:10s}: Threshold = {thresholds[c]:.2f}, Val Macro F₀.₅ = {f05_scores[c]:.4f}")
    print(f"  Elapsed: {(time.time()-total_start)/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
