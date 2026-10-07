import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.config import *
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.evaluate import compute_f05_per_entity, parse_ground_truth
from src.train_ensemble import EnsemblePredictor
from src.fast_predict import soundex, STOPWORDS, COMMON_ADDR_STOP
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

print("=" * 70)
print("TESTING THRESHOLD & BLOCKING IMPACT ON GROUND TRUTH")
print("=" * 70)

# Load ensemble models
with open('output/models_ensemble.pkl', 'rb') as f:
    saved = pickle.load(f)
models = saved['models']

gt = parse_ground_truth(TRAIN_GT)

for country in ['US', 'India']:
    print(f"\n{'='*50}")
    print(f"EVALUATING {country}")
    print(f"{'='*50}")

    model = models[country]
    is_india = (country == 'India')

    # Load 3,000 S1 records for this country
    s1_recs = []
    with open(TRAIN_S1, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 3 and parts[3].strip() == country:
                s1_recs.append(normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', country))
                if len(s1_recs) >= 3000:
                    break

    s1_ids = {r['entity_id'] for r in s1_recs}
    needed_matches = set()
    for r in s1_recs:
        needed_matches.update(gt.get(r['entity_id'], set()))

    print(f"Loaded {len(s1_recs):,} S1 entities. True matches needed: {len(needed_matches):,}")
    true_singletons_cnt = sum(1 for r in s1_recs if len(gt.get(r['entity_id'], set())) == 0)
    print(f"True singletons in sample: {true_singletons_cnt:,} ({true_singletons_cnt/len(s1_recs)*100:.1f}%)")

    # Load pool records: all needed matches + 100,000 distractors
    import random
    random.seed(42)
    pool_records = []
    for path in [TRAIN_S2, TRAIN_S3]:
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 3 and parts[3].strip() == country:
                    eid = parts[0]
                    if eid in needed_matches:
                        pool_records.append(normalize_record(eid, parts[1], parts[2] if len(parts) > 2 else '', country))
                    elif len(pool_records) < 150000 and random.random() < 0.05:
                        pool_records.append(normalize_record(eid, parts[1], parts[2] if len(parts) > 2 else '', country))

    print(f"Pool size: {len(pool_records):,} records (contains all {len(needed_matches):,} true matches)")

    # Build dual indexes with NO cap on tokens (or cap=3000)
    s2_name_idx = defaultdict(list)
    s2_pref_idx = defaultdict(list)
    s2_num_idx = defaultdict(list)
    s2_addr_idx = defaultdict(list)
    s2_sx_idx = defaultdict(list)

    s3_name_idx = defaultdict(list)
    s3_pref_idx = defaultdict(list)
    s3_num_idx = defaultdict(list)
    s3_addr_idx = defaultdict(list)
    s3_sx_idx = defaultdict(list)

    MAX_LEN = 3000

    for pi, rec in enumerate(pool_records):
        is_s2 = rec['entity_id'].startswith('S2-')
        name_idx = s2_name_idx if is_s2 else s3_name_idx
        pref_idx = s2_pref_idx if is_s2 else s3_pref_idx
        num_idx = s2_num_idx if is_s2 else s3_num_idx
        addr_idx = s2_addr_idx if is_s2 else s3_addr_idx
        sx_idx = s2_sx_idx if is_s2 else s3_sx_idx

        for tok in rec['name_tokens']:
            if len(tok) >= 3 and tok not in STOPWORDS and len(name_idx[tok]) < MAX_LEN:
                name_idx[tok].append(pi)

        if len(rec['norm_name']) >= 4 and len(pref_idx[rec['norm_name'][:4]]) < MAX_LEN:
            pref_idx[rec['norm_name'][:4]].append(pi)

        for num in rec['addr_numbers']:
            if len(num) >= 2 and len(num_idx[num]) < MAX_LEN:
                num_idx[num].append(pi)

        for tok in rec['norm_addr'].split():
            if len(tok) >= 5 and tok not in COMMON_ADDR_STOP and len(addr_idx[tok]) < MAX_LEN:
                addr_idx[tok].append(pi)

        if is_india:
            for tok in rec['name_tokens']:
                if len(tok) >= 3 and tok not in STOPWORDS:
                    sx = soundex(tok)
                    if sx and len(sx_idx[sx]) < MAX_LEN:
                        sx_idx[sx].append(pi)

    from src.fast_predict import query_source_candidates
    s2_idxs = (s2_name_idx, s2_pref_idx, s2_num_idx, s2_addr_idx, s2_sx_idx)
    s3_idxs = (s3_name_idx, s3_pref_idx, s3_num_idx, s3_addr_idx, s3_sx_idx)

    # Featurize pairs in batch
    batch_pairs = []
    pair_meta = []  # (s1_idx, pool_idx)
    cand_recall_hits = 0
    total_true_pairs = 0

    for s1_idx, s1_rec in enumerate(s1_recs):
        true_set = gt.get(s1_rec['entity_id'], set())
        total_true_pairs += len(true_set)

        c2 = query_source_candidates(s1_rec, s2_idxs, is_india, top_k=25)
        c3 = query_source_candidates(s1_rec, s3_idxs, is_india, top_k=25)
        pool_indices = c2 + c3
        cand_eids = {pool_records[pi]['entity_id'] for pi in pool_indices}

        for tm in true_set:
            if tm in cand_eids:
                cand_recall_hits += 1

        for pi in pool_indices:
            feats = compute_features(s1_rec, pool_records[pi])
            batch_pairs.append([feats[fn] for fn in FEATURE_NAMES])
            pair_meta.append((s1_idx, pi))

    print(f"Candidate Recall: {cand_recall_hits}/{total_true_pairs} ({cand_recall_hits/total_true_pairs*100:.2f}%)")
    print(f"Evaluating {len(batch_pairs):,} pairs with ensemble model...")

    X = np.array(batch_pairs, dtype=np.float32)
    probs = model.predict(X)

    # Sweep thresholds and compute Macro F0.5
    print(f"\n--- Threshold Sweep on Macro F0.5 ({country}) ---")
    best_th = 0.5
    best_f05 = 0.0

    for th in [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]:
        preds = defaultdict(set)
        for (s1_idx, pi), prob in zip(pair_meta, probs):
            if prob >= th:
                preds[s1_recs[s1_idx]['entity_id']].add(pool_records[pi]['entity_id'])

        # Compute Macro F0.5
        scores = []
        singleton_cnt = 0
        for s1_rec in s1_recs:
            sid = s1_rec['entity_id']
            p_set = preds.get(sid, set())
            t_set = gt.get(sid, set())
            if len(p_set) == 0:
                singleton_cnt += 1
            scores.append(compute_f05_per_entity(p_set, t_set))

        f05 = np.mean(scores)
        sing_pct = singleton_cnt / len(s1_recs) * 100
        print(f"  Threshold {th:.2f} -> Macro F0.5 = {f05:.4f} | Singletons = {singleton_cnt:,} ({sing_pct:.1f}%)")
        if f05 > best_f05:
            best_f05 = f05
            best_th = th

    print(f"\n🏆 Best threshold for {country}: {best_th:.2f} (Macro F0.5 = {best_f05:.4f})")
