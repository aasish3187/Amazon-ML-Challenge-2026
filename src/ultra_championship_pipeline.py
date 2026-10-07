"""
ULTRA CHAMPIONSHIP PREDICTION PIPELINE: Amazon ML Challenge 2026
================================================================
Key Innovations:
1. Dual Independent S2 & S3 Inverted Indexes (Zero Candidate Starvation)
2. IDF-Weighted Multi-Key Scoring (Name tokens, Soundex phonetics, Numeric tokens, Address tokens, 4-char prefix)
3. High-Efficiency Candidate Gating (Top-15 per source with score >= 1.5)
4. Gated Tri-Ensemble ML Scoring (LightGBM vector filter + XGBoost + CatBoost ensemble average)
5. Precision Guard (High calibrated threshold: France 0.82, US 0.85, India 0.82)
6. Transitive Graph Triangle Closure (Recovers dual S2+S3 matches)
7. Resumable Checkpoint Execution (Saves progress, skips completed entities)
8. Strict S1 Entity Alignment to test_source1.tsv order
9. Automated Packaging & Validation via student_resource/utils/validate_submission.py
"""

import os
import sys
import time
import pickle
import math
import subprocess
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from src.config import *
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.fast_predict import soundex
from src.train_ensemble import EnsemblePredictor
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

STOPWORDS = {
    'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd',
    'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions',
    'enterprises', 'international', 'holdings', 'global', 'technologies',
    'sarl', 'sasu', 'eurl', 'association'
}

COMMON_ADDR = {
    'road', 'street', 'lane', 'floor', 'building', 'near', 'opposite',
    'behind', 'beside', 'plot', 'shop', 'flat', 'apartment', 'nagar',
    'colony', 'india', 'state', 'city', 'cross', 'main', 'sector',
    'delhi', 'mumbai', 'bengaluru', 'hyderabad', 'chennai', 'kolkata'
}

EXPECTED_COUNTS = {
    'France': 259452,
    'US': 663106,
    'India': 809986
}

def build_source_idf_index(pool_records):
    """Build an IDF-weighted inverted index for a single source (S2 or S3)."""
    N = len(pool_records)
    name_df = Counter()
    addr_df = Counter()
    num_df = Counter()
    sx_df = Counter()
    pref_df = Counter()
    
    for rec in pool_records:
        for t in rec['name_tokens']:
            if len(t) >= 3 and t not in STOPWORDS:
                name_df[t] += 1
                sx = soundex(t)
                if sx: sx_df[sx] += 1
        for num in rec['addr_numbers']:
            num_df[num] += 1
        for t in rec['norm_addr'].split():
            if len(t) >= 3 and t not in COMMON_ADDR:
                addr_df[t] += 1
        if len(rec['norm_name']) >= 4:
            pref_df[rec['norm_name'][:4]] += 1
            
    def idf(df):
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
    name_idx = defaultdict(list)
    num_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    sx_idx = defaultdict(list)
    pref_idx = defaultdict(list)
    
    for idx, rec in enumerate(pool_records):
        for t in rec['name_tokens']:
            if len(t) >= 3 and t not in STOPWORDS and name_df[t] < 0.05 * N:
                name_idx[t].append(idx)
            sx = soundex(t)
            if sx and sx_df[sx] < 0.05 * N:
                sx_idx[sx].append(idx)
        for num in rec['addr_numbers']:
            if num_df[num] < 0.05 * N:
                num_idx[num].append(idx)
        for t in rec['norm_addr'].split():
            if len(t) >= 3 and t not in COMMON_ADDR and addr_df[t] < 0.05 * N:
                addr_idx[t].append(idx)
        if len(rec['norm_name']) >= 4 and pref_df[rec['norm_name'][:4]] < 0.05 * N:
            pref_idx[rec['norm_name'][:4]].append(idx)
            
    return (name_idx, num_idx, addr_idx, sx_idx, pref_idx), (name_df, num_df, addr_df, sx_df, pref_df), idf


def query_top_k(s1_rec, indexes, dfs, idf_fn, top_k=15):
    """Retrieve top-K candidates using multi-key IDF scoring with noise floor."""
    name_idx, num_idx, addr_idx, sx_idx, pref_idx = indexes
    name_df, num_df, addr_df, sx_df, pref_df = dfs
    scores = defaultdict(float)
    
    for t in s1_rec['name_tokens']:
        if t in name_idx:
            sc = idf_fn(name_df[t]) * 3.0
            for pi in name_idx[t][:1200]:
                scores[pi] += sc
        sx = soundex(t)
        if sx in sx_idx:
            sc = idf_fn(sx_df[sx]) * 1.5
            for pi in sx_idx[sx][:600]:
                scores[pi] += sc
                
    for num in s1_rec['addr_numbers']:
        if num in num_idx:
            w = 4.0 if len(num) == 6 else 1.5
            sc = idf_fn(num_df[num]) * w
            for pi in num_idx[num][:1200]:
                scores[pi] += sc
                
    for t in s1_rec['norm_addr'].split():
        if t in addr_idx:
            sc = idf_fn(addr_df[t]) * 1.5
            for pi in addr_idx[t][:1200]:
                scores[pi] += sc
                
    if len(s1_rec['norm_name']) >= 4:
        pref = s1_rec['norm_name'][:4]
        if pref in pref_idx:
            sc = idf_fn(pref_df[pref]) * 1.2
            for pi in pref_idx[pref][:600]:
                scores[pi] += sc
                
    if not scores:
        return []
    # Filter candidates with score >= 1.5 to eliminate low-overlap noise
    filtered = [(pi, s) for pi, s in scores.items() if s >= 1.5]
    filtered.sort(key=lambda x: -x[1])
    return [pi for pi, _ in filtered[:top_k]]


def merge_and_validate():
    """Merge current country files into final matching_results.tsv and candidate_pairs.tsv."""
    countries = ['France', 'US', 'India']
    preds_map = {}
    cands_map = {}
    
    for country in countries:
        matching_ckpt = os.path.join(OUTPUT_DIR, f"matching_{country}.tsv")
        candidate_ckpt = os.path.join(OUTPUT_DIR, f"candidate_{country}.tsv")
        
        if os.path.exists(matching_ckpt):
            with open(matching_ckpt, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    preds_map[parts[0]] = parts[1] if len(parts) > 1 else ''
                    
        if os.path.exists(candidate_ckpt):
            with open(candidate_ckpt, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    cands_map[parts[0]] = parts[1] if len(parts) > 1 else ''
                    
    print(f"\n  [Checkpoint Merge] Writing {len(preds_map):,} entities in test_source1.tsv order...")
    matching_final = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_final = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    with open(matching_final, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        with open(TEST_S1, 'r', encoding='utf-8') as f_in:
            next(f_in)
            for line in f_in:
                eid = line.strip().split('\t')[0]
                m = preds_map.get(eid, '')
                f_out.write(f"{eid}\t{m}\n")
                
    with open(candidate_final, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        with open(TEST_S1, 'r', encoding='utf-8') as f_in:
            next(f_in)
            for line in f_in:
                eid = line.strip().split('\t')[0]
                c = cands_map.get(eid, '')
                f_out.write(f"{eid}\t{c}\n")
                
    print(f"  Generated {matching_final} ({os.path.getsize(matching_final)/1e6:.1f} MB)")
    print(f"  Generated {candidate_final} ({os.path.getsize(candidate_final)/1e6:.1f} MB)")
    
    # Run validation
    cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", matching_final,
        "--candidate", candidate_final,
        "--test-dir", "student_resource/dataset/test"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if "PASS" in res.stdout:
        print("  ✅ VALIDATION STATUS: PASS (100% Valid Submission)")
    else:
        print("  Validator output:\n", res.stdout)


def process_country(country: str, model, threshold: float, top_k_per_source: int = 15, chunk_size: int = 15000):
    """Stream inference for an entire country partition with checkpoint resuming."""
    partition_dir = f"partitions/test/{country}"
    pool_path = os.path.join(partition_dir, "pool.tsv")
    s1_path = os.path.join(partition_dir, "s1.tsv")
    matching_ckpt = os.path.join(OUTPUT_DIR, f"matching_{country}.tsv")
    candidate_ckpt = os.path.join(OUTPUT_DIR, f"candidate_{country}.tsv")
    expected_total = EXPECTED_COUNTS.get(country, 0)
    
    print(f"\n{'='*70}")
    print(f"  PROCESSING COUNTRY: {country} (Threshold = {threshold:.2f}, Top-K = {top_k_per_source})")
    print(f"{'='*70}")
    
    # Check already processed records
    already_done = set()
    initial_matched = 0
    initial_singletons = 0
    if os.path.exists(matching_ckpt) and os.path.exists(candidate_ckpt):
        with open(matching_ckpt, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = line.strip().split('\t')
                if parts[0]:
                    already_done.add(parts[0])
                    if len(parts) > 1 and parts[1].strip():
                        initial_matched += 1
                    else:
                        initial_singletons += 1
                        
    if len(already_done) >= expected_total:
        print(f"  Country {country} already 100% complete ({len(already_done):,} entities). Skipping!")
        return
        
    resuming = len(already_done) > 0
    if resuming:
        print(f"  Resuming {country} from entity {len(already_done):,} / {expected_total:,} "
              f"({len(already_done)/expected_total*100:.1f}% already saved)...")
        matching_out = open(matching_ckpt, 'a', encoding='utf-8')
        candidate_out = open(candidate_ckpt, 'a', encoding='utf-8')
    else:
        matching_out = open(matching_ckpt, 'w', encoding='utf-8')
        candidate_out = open(candidate_ckpt, 'w', encoding='utf-8')
        matching_out.write("source1_entity_id\tmatched_entity_ids\n")
        candidate_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
    t_start = time.time()
    
    # 1. Load pool and split into S2 and S3
    print(f"  [1/3] Loading pool records from {pool_path}...")
    s2_records = []
    s3_records = []
    with open(pool_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3])
            if parts[0].startswith('S2-'):
                s2_records.append(rec)
            else:
                s3_records.append(rec)
                
    print(f"  Loaded {len(s2_records):,} S2 records and {len(s3_records):,} S3 records in {time.time()-t_start:.1f}s")
    
    # 2. Build independent indexes
    print(f"  Building independent S2 and S3 IDF indexes...")
    t_idx = time.time()
    s2_idx, s2_df, s2_idf = build_source_idf_index(s2_records)
    s3_idx, s3_df, s3_idf = build_source_idf_index(s3_records)
    print(f"  Indexes built in {time.time()-t_idx:.1f}s")
    
    # 3. Stream S1 and run inference in chunks
    print(f"  [2/3] Streaming S1 and predicting matches...")
    total_s1 = len(already_done)
    total_matched_s1 = initial_matched
    total_singletons = initial_singletons
    total_transitive = 0
    t_stream_start = time.time()
    
    chunk_s1 = []
    
    def process_chunk(s1_batch):
        nonlocal total_s1, total_matched_s1, total_singletons, total_transitive
        batch_pairs = []
        batch_meta = [] # (s1_local_idx, cid, rec)
        s1_cand_ids_map = defaultdict(list)
        s1_cand_recs_map = defaultdict(list)
        
        for s1_local_idx, s1_rec in enumerate(s1_batch):
            c2_pis = query_top_k(s1_rec, s2_idx, s2_df, s2_idf, top_k=top_k_per_source)
            c3_pis = query_top_k(s1_rec, s3_idx, s3_df, s3_idf, top_k=top_k_per_source)
            
            cands_all = []
            for pi in c2_pis:
                rec = s2_records[pi]
                cands_all.append(rec['entity_id'])
                s1_cand_recs_map[s1_local_idx].append(rec)
                feats = compute_features(s1_rec, rec)
                batch_pairs.append([feats[fn] for fn in FEATURE_NAMES])
                batch_meta.append((s1_local_idx, rec['entity_id'], rec))
                
            for pi in c3_pis:
                rec = s3_records[pi]
                cands_all.append(rec['entity_id'])
                s1_cand_recs_map[s1_local_idx].append(rec)
                feats = compute_features(s1_rec, rec)
                batch_pairs.append([feats[fn] for fn in FEATURE_NAMES])
                batch_meta.append((s1_local_idx, rec['entity_id'], rec))
                
            s1_cand_ids_map[s1_local_idx] = sorted(list(set(cands_all)))
            
        if batch_pairs:
            X = np.array(batch_pairs, dtype=np.float32)
            # Stage 1: Fast LightGBM vector evaluation (C++ OpenMP)
            lgb_probs = model.lgb_model.predict(X)
            
            # Stage 2: Lossless Gated Tri-Ensemble for ambiguous/positive candidates
            candidate_mask = lgb_probs >= 0.20
            if np.any(candidate_mask):
                sub_X = X[candidate_mask]
                p_xgb = model.xgb_model.predict_proba(sub_X)[:, 1]
                p_cb = model.cb_model.predict_proba(sub_X)[:, 1]
                probs = lgb_probs.copy()
                probs[candidate_mask] = (
                    model.weights[0] * lgb_probs[candidate_mask] +
                    model.weights[1] * p_xgb +
                    model.weights[2] * p_cb
                )
            else:
                probs = lgb_probs
        else:
            probs = np.array([], dtype=np.float32)
            
        # Group predictions by s1_local_idx
        s1_probs_map = defaultdict(list)
        for (s1_local_idx, cid, rec), prob in zip(batch_meta, probs):
            s1_probs_map[s1_local_idx].append((cid, prob, rec))
            
        # Decide matches for each S1 entity
        for s1_local_idx, s1_rec in enumerate(s1_batch):
            eid = s1_rec['entity_id']
            c_list = s1_probs_map.get(s1_local_idx, [])
            cand_ids = s1_cand_ids_map.get(s1_local_idx, [])
            
            matched = {cid for cid, prob, _ in c_list if prob >= threshold}
            
            # Transitive Triangle Closure:
            # If S1 matched S2 with high confidence (prob >= 0.85),
            # check if any S3 candidate in candidate pool is identical to that S2
            s2_high = [rec for cid, prob, rec in c_list if prob >= 0.85 and cid.startswith('S2-')]
            s3_cands = [(cid, rec) for cid, prob, rec in c_list if cid.startswith('S3-') and cid not in matched]
            for s2_rec in s2_high:
                for s3_cid, s3_rec in s3_cands:
                    if s2_rec['norm_name'] == s3_rec['norm_name'] and (not s2_rec['norm_addr'] or not s3_rec['norm_addr'] or s2_rec['norm_addr'] == s3_rec['norm_addr']):
                        matched.add(s3_cid)
                        total_transitive += 1
                        
            matched_sorted = sorted(list(matched))
            # Ensure all matched IDs are in candidate set for 100% strict validator compliance
            cand_ids_all = sorted(list(set(cand_ids) | matched))
            
            matching_out.write(f"{eid}\t{','.join(matched_sorted)}\n")
            candidate_out.write(f"{eid}\t{','.join(cand_ids_all)}\n")
            
            total_s1 += 1
            if matched_sorted:
                total_matched_s1 += 1
            else:
                total_singletons += 1
                
        matching_out.flush()
        candidate_out.flush()
        
    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            if parts[0] in already_done:
                continue
            rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3])
            chunk_s1.append(rec)
            if len(chunk_s1) >= chunk_size:
                process_chunk(chunk_s1)
                chunk_s1 = []
                elapsed_stream = time.time() - t_stream_start
                records_streamed = total_s1 - len(already_done)
                rate = records_streamed / elapsed_stream if elapsed_stream > 0 else 0
                pct = total_s1 / expected_total * 100 if expected_total > 0 else 0
                eta_sec = (expected_total - total_s1) / rate if rate > 0 else 0
                print(f"    Processed {total_s1:,} / {expected_total:,} S1 ({pct:.1f}%) — "
                      f"{total_matched_s1:,} matched, {total_singletons:,} singletons [{total_singletons/total_s1*100:.1f}%] — "
                      f"{rate:.0f} S1/sec (ETA: {eta_sec/60:.1f} min)")
                      
        if chunk_s1:
            process_chunk(chunk_s1)
            
    matching_out.close()
    candidate_out.close()
    
    elapsed = time.time() - t_start
    print(f"\n  Country {country} complete in {elapsed/60:.1f} minutes!")
    print(f"    Total S1: {total_s1:,} | Matched: {total_matched_s1:,} ({total_matched_s1/total_s1*100:.1f}%) | "
          f"Singletons: {total_singletons:,} ({total_singletons/total_s1*100:.1f}%)")
    print(f"    Transitive closures: {total_transitive:,}")
    
    del s2_records, s3_records, s2_idx, s3_idx
    import gc
    gc.collect()
    
    # Checkpoint merge and validation immediately after this country
    merge_and_validate()


def main():
    t_start = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — CHAMPIONSHIP PREDICTION ENGINE")
    print("=" * 70)
    
    model_path = os.path.join(OUTPUT_DIR, 'models_ensemble.pkl')
    print(f"Loading ensemble models from {model_path}...")
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    models = saved['models']
    
    thresholds = {
        'France': 0.82,
        'US': 0.85,
        'India': 0.82
    }
    
    # Process countries in optimal order: France, US, India
    countries = ['France', 'US', 'India']
    for country in countries:
        model = models[country] if country in models else models['US']
        thresh = thresholds.get(country, 0.82)
        process_country(country, model, thresh, top_k_per_source=15, chunk_size=15000)
        
    print(f"\n[Finalizing] Creating final submission package...")
    from src.build_final_submission_package import build_package
    build_package()
    
    print(f"\n🏆 CHAMPIONSHIP PIPELINE FINISHED IN {(time.time()-t_start)/60:.1f} MINUTES!")


if __name__ == '__main__':
    main()
