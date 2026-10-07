"""
End-to-End Entity Resolution Pipeline.
Orchestrates: Load → Normalize → Partition → Block → Featurize → Classify → Output.

Usage:
    python src/pipeline.py --mode train    # Train on training data, evaluate on validation
    python src/pipeline.py --mode test     # Generate test predictions
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.normalize import normalize_record
from src.evaluate import compute_macro_f05, compute_f05_per_entity, parse_ground_truth
from src.features import compute_features, FEATURE_NAMES
from src.blocking import run_full_blocking


def load_source_streaming(filepath: str, max_rows: int = None) -> list:
    """
    Load a source TSV and normalize records via streaming.
    Returns list of normalized record dicts.
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        id_idx = header.index('entity_id')
        name_idx = header.index('business_name')
        addr_idx = header.index('business_address')
        country_idx = header.index('country')

        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            parts = line.strip().split('\t')
            if len(parts) <= max(id_idx, name_idx, addr_idx, country_idx):
                continue
            rec = normalize_record(
                parts[id_idx],
                parts[name_idx],
                parts[addr_idx] if addr_idx < len(parts) else '',
                parts[country_idx] if country_idx < len(parts) else '',
            )
            records.append(rec)

            if (i + 1) % 500000 == 0:
                print(f"    Loaded {i+1:,} records...")

    print(f"    Loaded {len(records):,} records from {os.path.basename(filepath)}")
    return records


def partition_by_country(records: list) -> dict:
    """Split records by country → dict of country → list of records."""
    partitions = defaultdict(list)
    for rec in records:
        partitions[rec['country']].append(rec)
    for country, recs in partitions.items():
        print(f"    {country}: {len(recs):,} records")
    return dict(partitions)


def build_training_pairs(s1_records: list, pool_records: list,
                         blocking_results: dict,
                         gt_map: dict, max_negatives_per_positive: int = 5):
    """
    Build labeled training pairs from blocking results + ground truth.

    Returns:
        X: np.array of shape (n_pairs, n_features)
        y: np.array of shape (n_pairs,)  — 1 for match, 0 for non-match
        pair_ids: list of (s1_id, candidate_id) tuples
    """
    print(f"\n  Building training pairs...")
    t0 = time.time()

    pool_id_to_idx = {r['entity_id']: i for i, r in enumerate(pool_records)}
    X_rows = []
    y_rows = []
    pair_ids = []

    for s1_idx, pool_indices in blocking_results.items():
        s1_rec = s1_records[s1_idx]
        s1_id = s1_rec['entity_id']
        true_matches = gt_map.get(s1_id, set())

        for pool_idx in pool_indices:
            pool_rec = pool_records[pool_idx]
            pool_id = pool_rec['entity_id']

            feats = compute_features(s1_rec, pool_rec)
            feat_values = [feats[fn] for fn in FEATURE_NAMES]

            label = 1 if pool_id in true_matches else 0

            X_rows.append(feat_values)
            y_rows.append(label)
            pair_ids.append((s1_id, pool_id))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"    Total pairs: {len(y):,} (positive: {n_pos:,}, negative: {n_neg:,})")
    print(f"    Built in {time.time()-t0:.1f}s")

    return X, y, pair_ids


def train_classifier(X_train, y_train, X_val, y_val):
    """Train LightGBM classifier with early stopping."""
    import lightgbm as lgb

    print(f"\n  Training LightGBM...")
    t0 = time.time()

    train_data = lgb.Dataset(X_train, y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, y_val, feature_name=FEATURE_NAMES, reference=train_data)

    params = LIGHTGBM_PARAMS.copy()
    n_iter = params.pop('num_iterations', 500)
    early = params.pop('early_stopping_rounds', 30)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_iter,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early),
            lgb.log_evaluation(period=50),
        ],
    )

    print(f"    Trained in {time.time()-t0:.1f}s, best iteration: {model.best_iteration}")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    print(f"\n    Feature Importance (top 10):")
    for fname, imp in feat_imp[:10]:
        print(f"      {fname:30s} {imp:10.1f}")

    return model


def optimize_threshold(model, X_val, y_val, val_pair_ids, gt_map):
    """
    Search for the threshold that maximizes macro F₀.₅ on validation data.
    """
    print(f"\n  Optimizing threshold for F₀.₅...")
    scores = model.predict(X_val)

    # Group predictions by S1 entity
    s1_preds = defaultdict(list)  # s1_id → list of (score, candidate_id)
    for (s1_id, cand_id), score in zip(val_pair_ids, scores):
        s1_preds[s1_id].append((score, cand_id))

    # Collect all S1 IDs that should be evaluated
    all_s1_ids = set(gt_map.keys())

    best_threshold = 0.5
    best_f05 = 0.0

    for threshold in np.arange(
        THRESHOLD_SEARCH_RANGE[0],
        THRESHOLD_SEARCH_RANGE[1],
        THRESHOLD_SEARCH_STEP
    ):
        predictions = {}
        for s1_id in all_s1_ids:
            matched = set()
            for score, cand_id in s1_preds.get(s1_id, []):
                if score >= threshold:
                    matched.add(cand_id)
            predictions[s1_id] = matched

        f05 = compute_macro_f05(predictions, gt_map)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    print(f"    Best threshold: {best_threshold:.2f}")
    print(f"    Best macro F₀.₅: {best_f05:.4f}")

    return best_threshold, best_f05


def generate_predictions(model, s1_records, pool_records, blocking_results,
                         threshold):
    """
    Generate final predictions using the trained model and threshold.
    Returns dict: s1_id → set of matched candidate IDs
    """
    print(f"\n  Generating predictions (threshold={threshold:.2f})...")
    t0 = time.time()

    predictions = {}
    candidates_dict = {}

    for s1_idx in range(len(s1_records)):
        s1_rec = s1_records[s1_idx]
        s1_id = s1_rec['entity_id']
        pool_indices = blocking_results.get(s1_idx, set())

        matched = set()
        all_candidates = set()

        if pool_indices:
            for pool_idx in pool_indices:
                pool_rec = pool_records[pool_idx]
                pool_id = pool_rec['entity_id']
                all_candidates.add(pool_id)

                feats = compute_features(s1_rec, pool_rec)
                feat_values = [[feats[fn] for fn in FEATURE_NAMES]]
                score = model.predict(np.array(feat_values, dtype=np.float32))[0]

                if score >= threshold:
                    matched.add(pool_id)

        predictions[s1_id] = matched
        candidates_dict[s1_id] = all_candidates

    print(f"    Generated predictions for {len(predictions):,} S1 entities in {time.time()-t0:.1f}s")

    n_matched = sum(1 for v in predictions.values() if len(v) > 0)
    n_singleton = sum(1 for v in predictions.values() if len(v) == 0)
    print(f"    Matched: {n_matched:,}, Singletons: {n_singleton:,}")

    return predictions, candidates_dict


def write_output(predictions: dict, candidates: dict,
                 matching_path: str, candidate_path: str,
                 all_s1_ids: list):
    """Write matching_results.tsv and candidate_pairs.tsv."""
    os.makedirs(os.path.dirname(matching_path), exist_ok=True)

    print(f"\n  Writing {matching_path}...")
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = predictions.get(s1_id, set())
            matched_str = ','.join(sorted(matched))
            f.write(f"{s1_id}\t{matched_str}\n")

    print(f"  Writing {candidate_path}...")
    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, set())
            cands_str = ','.join(sorted(cands))
            f.write(f"{s1_id}\t{cands_str}\n")

    print(f"  ✅ Output written!")


def run_country_pipeline(country, s1_recs, pool_recs, gt_map=None,
                         mode='train', model=None, threshold=0.5):
    """
    Run the full pipeline for a single country.

    In 'train' mode: blocks, featurizes, trains classifier, evaluates.
    In 'test' mode: blocks, featurizes, predicts using provided model.

    Returns: (predictions, candidates, model, threshold)
    """
    print(f"\n{'='*70}")
    print(f"  PROCESSING COUNTRY: {country} ({len(s1_recs):,} S1, {len(pool_recs):,} pool)")
    print(f"{'='*70}")

    # Stage 3: Blocking
    blocking_results = run_full_blocking(
        s1_recs, pool_recs,
        top_k=TFIDF_TOP_K,
        max_candidates=MAX_CANDIDATES_PER_S1,
    )

    if mode == 'train' and gt_map:
        # Measure blocking recall
        total_true_matches = 0
        found_in_blocking = 0
        pool_id_to_idx = {r['entity_id']: i for i, r in enumerate(pool_recs)}
        for s1_idx, s1_rec in enumerate(s1_recs):
            s1_id = s1_rec['entity_id']
            true_matches = gt_map.get(s1_id, set())
            pool_indices = blocking_results.get(s1_idx, set())
            candidate_ids = {pool_recs[pi]['entity_id'] for pi in pool_indices}
            for tm in true_matches:
                total_true_matches += 1
                if tm in candidate_ids:
                    found_in_blocking += 1

        blocking_recall = found_in_blocking / total_true_matches if total_true_matches > 0 else 0
        print(f"\n  Blocking Recall: {blocking_recall:.4f} "
              f"({found_in_blocking:,}/{total_true_matches:,})")

        # Split into train/val by S1 indices
        n = len(s1_recs)
        n_val = max(1, int(n * VALIDATION_FRACTION))
        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(n)
        val_indices = set(indices[:n_val])
        train_indices = set(indices[n_val:])

        # Build training pairs (from train split only)
        train_blocking = {i: blocking_results.get(i, set()) for i in train_indices}
        val_blocking = {i: blocking_results.get(i, set()) for i in val_indices}

        train_gt = {s1_recs[i]['entity_id']: gt_map.get(s1_recs[i]['entity_id'], set())
                    for i in train_indices}
        val_gt = {s1_recs[i]['entity_id']: gt_map.get(s1_recs[i]['entity_id'], set())
                  for i in val_indices}

        X_train, y_train, train_pairs = build_training_pairs(
            s1_recs, pool_recs, train_blocking, gt_map)
        X_val, y_val, val_pairs = build_training_pairs(
            s1_recs, pool_recs, val_blocking, gt_map)

        if len(X_train) == 0 or y_train.sum() == 0:
            print("  ⚠️  No positive training pairs found! Skipping classifier.")
            return {}, {}, None, threshold

        # Stage 5: Train classifier
        model = train_classifier(X_train, y_train, X_val, y_val)

        # Stage 6: Optimize threshold
        threshold, f05 = optimize_threshold(model, X_val, y_val, val_pairs, val_gt)

        # Generate full predictions for evaluation
        preds, cands = generate_predictions(model, s1_recs, pool_recs,
                                            blocking_results, threshold)

        # Evaluate on validation set
        val_preds = {s1_id: preds.get(s1_id, set()) for s1_id in val_gt}
        from src.evaluate import evaluate_from_dicts
        print(f"\n  Validation Results ({country}):")
        evaluate_from_dicts(val_preds, val_gt, verbose=True)

        return preds, cands, model, threshold

    elif mode == 'test' and model is not None:
        # Inference mode
        preds, cands = generate_predictions(model, s1_recs, pool_recs,
                                            blocking_results, threshold)
        return preds, cands, model, threshold

    else:
        raise ValueError(f"Invalid mode={mode} or missing model/gt_map")


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    parser.add_argument('--mode', choices=['train', 'test', 'full'],
                        default='train',
                        help="'train': train+eval, 'test': generate test output, "
                             "'full': train then test")
    parser.add_argument('--max-rows', type=int, default=None,
                        help="Limit rows per source file (for debugging)")
    args = parser.parse_args()

    total_start = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — ENTITY RESOLUTION PIPELINE")
    print(f"  Mode: {args.mode}")
    print("=" * 70)

    # ─── Load Data ────────────────────────────────────────────────────────
    if args.mode in ('train', 'full'):
        print("\n📥 Loading training data...")
        s1_train = load_source_streaming(TRAIN_S1, args.max_rows)
        s2_train = load_source_streaming(TRAIN_S2, args.max_rows)
        s3_train = load_source_streaming(TRAIN_S3, args.max_rows)
        pool_train = s2_train + s3_train
        del s2_train, s3_train  # Free memory

        gt_map = parse_ground_truth(TRAIN_GT)
        print(f"    Ground truth: {len(gt_map):,} S1 entities")

        # Partition by country
        print("\n🌍 Partitioning by country...")
        s1_by_country = partition_by_country(s1_train)
        pool_by_country = partition_by_country(pool_train)
        del s1_train, pool_train

        # Run pipeline per country
        all_models = {}
        all_thresholds = {}
        all_preds = {}
        all_cands = {}

        for country in sorted(s1_by_country.keys()):
            if country not in pool_by_country:
                print(f"\n  ⚠️  Country '{country}' has no pool records, skipping.")
                continue

            preds, cands, model, thresh = run_country_pipeline(
                country,
                s1_by_country[country],
                pool_by_country[country],
                gt_map=gt_map,
                mode='train',
            )

            all_models[country] = model
            all_thresholds[country] = thresh
            all_preds.update(preds)
            all_cands.update(cands)

        # Save models
        model_path = os.path.join(OUTPUT_DIR, 'models.pkl')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({'models': all_models, 'thresholds': all_thresholds}, f)
        print(f"\n  💾 Models saved to {model_path}")

    if args.mode in ('test', 'full'):
        print("\n📥 Loading test data...")
        s1_test = load_source_streaming(TEST_S1, args.max_rows)
        s2_test = load_source_streaming(TEST_S2, args.max_rows)
        s3_test = load_source_streaming(TEST_S3, args.max_rows)
        pool_test = s2_test + s3_test
        del s2_test, s3_test

        # Load models if in 'test' mode
        if args.mode == 'test':
            model_path = os.path.join(OUTPUT_DIR, 'models.pkl')
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            all_models = saved['models']
            all_thresholds = saved['thresholds']

        # Partition test data
        print("\n🌍 Partitioning test data by country...")
        s1_test_by_country = partition_by_country(s1_test)
        pool_test_by_country = partition_by_country(pool_test)
        del s1_test, pool_test

        all_test_preds = {}
        all_test_cands = {}

        for country in sorted(s1_test_by_country.keys()):
            if country not in pool_test_by_country:
                print(f"\n  ⚠️  Country '{country}' has no pool, predicting singletons.")
                for rec in s1_test_by_country[country]:
                    all_test_preds[rec['entity_id']] = set()
                    all_test_cands[rec['entity_id']] = set()
                continue

            # Use model from training; for France (unseen), use a fallback
            model = all_models.get(country)
            thresh = all_thresholds.get(country, DEFAULT_MATCH_THRESHOLD)

            if model is None:
                # Fallback for unseen countries (France): use US model or average
                fallback = all_models.get('US') or list(all_models.values())[0]
                if fallback:
                    print(f"\n  ℹ️  Using fallback model for '{country}'")
                    model = fallback
                    thresh = all_thresholds.get('US', DEFAULT_MATCH_THRESHOLD)
                else:
                    print(f"\n  ⚠️  No model available for '{country}', predicting singletons.")
                    for rec in s1_test_by_country[country]:
                        all_test_preds[rec['entity_id']] = set()
                        all_test_cands[rec['entity_id']] = set()
                    continue

            preds, cands, _, _ = run_country_pipeline(
                country,
                s1_test_by_country[country],
                pool_test_by_country[country],
                mode='test',
                model=model,
                threshold=thresh,
            )

            all_test_preds.update(preds)
            all_test_cands.update(cands)

        # Collect all test S1 IDs (including those with no predictions)
        all_test_s1_ids = []
        with open(TEST_S1, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                s1_id = line.strip().split('\t')[0]
                all_test_s1_ids.append(s1_id)

        # Write output
        write_output(all_test_preds, all_test_cands,
                     MATCHING_OUTPUT, CANDIDATE_OUTPUT, all_test_s1_ids)

        print(f"\n  📊 Final stats:")
        print(f"    Total S1 entities: {len(all_test_s1_ids):,}")
        n_matched = sum(1 for s in all_test_s1_ids if len(all_test_preds.get(s, set())) > 0)
        print(f"    Predicted matched: {n_matched:,}")
        print(f"    Predicted singletons: {len(all_test_s1_ids) - n_matched:,}")

    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ✅ PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
