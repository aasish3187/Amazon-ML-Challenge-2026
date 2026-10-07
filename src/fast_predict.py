"""
High-Performance, Championship Test Prediction Engine with Modular Checkpointing.
Features:
  - Dual Independent Inverted Indexes (S2 & S3 separately indexed, 0% starvation)
  - High-Capacity Indexing (MAX_INDEX_LEN=3000 to eliminate candidate starvation across 4.7M pool)
  - Phonetic Soundex Matching for Indian Transliterations
  - Deep Address & PIN code blocking
  - Tri-Ensemble Classification (LightGBM + XGBoost + CatBoost) with 28 Discriminative Features
  - Singleton-Rescuing Calibration (Eliminates false singletons while protecting true singletons)
  - Transitive Graph Triangle Closure (Recovers dual S2+S3 matches)
  - Modular Country Checkpointing (Crash-resilient, Resumes from existing country files)
  - Streaming Architecture with Bounded Memory & Strict File Ordering
"""
import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.train_ensemble import EnsemblePredictor
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

STOPWORDS = {
    'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd',
    'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions',
    'enterprises', 'international', 'holdings', 'global', 'technologies',
    'sarl', 'sasu', 'eurl', 'association'
}

COMMON_ADDR_STOP = {
    'the', 'and', 'road', 'rd', 'street', 'st', 'lane', 'ln', 'floor', 'fl',
    'building', 'bldg', 'near', 'opp', 'opposite', 'behind', 'beside', 'plot',
    'no', 'hno', 'dno', 'door', 'shop', 'flat', 'apartment', 'nagar', 'colony',
    'india', 'delhi', 'mumbai', 'bengaluru', 'hyderabad', 'chennai', 'kolkata',
    'state', 'city', 'west', 'east', 'north', 'south', 'cross', 'main', 'sector'
}

EXPECTED_COUNTS = {
    'France': 259452,
    'US': 663106,
    'India': 809986
}


def soundex(name: str) -> str:
    """Standard American Soundex algorithm for phonetic normalization."""
    if not name or not name.isalpha():
        return ''
    name = name.upper()
    codes = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first = name[0]
    tail = [codes.get(c, '') for c in name[1:]]
    res = [first]
    prev = codes.get(first, '')
    for c in tail:
        if c != prev and c != '':
            res.append(c)
        prev = c
    return (''.join(res) + '0000')[:4]


def build_country_dual_index(pool_tsv_path: str, country: str):
    """
    Stream and index a country's pool.tsv into memory with INDEPENDENT S2 and S3 indexes.
    Prevents Source 3 starvation, ensuring balanced candidate recall.
    """
    print(f"\n  [1/3] Loading and indexing pool from {pool_tsv_path}...")
    t0 = time.time()

    MAX_INDEX_LEN = 3000
    pool_records = []

    # Independent indexes for S2
    s2_name_idx = defaultdict(list)
    s2_pref_idx = defaultdict(list)
    s2_num_idx = defaultdict(list)
    s2_addr_idx = defaultdict(list)
    s2_sx_idx = defaultdict(list)

    # Independent indexes for S3
    s3_name_idx = defaultdict(list)
    s3_pref_idx = defaultdict(list)
    s3_num_idx = defaultdict(list)
    s3_addr_idx = defaultdict(list)
    s3_sx_idx = defaultdict(list)

    is_india = (country == 'India')

    with open(pool_tsv_path, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3])
            pool_idx = len(pool_records)
            pool_records.append(rec)

            is_s2 = parts[0].startswith('S2-')
            name_idx = s2_name_idx if is_s2 else s3_name_idx
            pref_idx = s2_pref_idx if is_s2 else s3_pref_idx
            num_idx = s2_num_idx if is_s2 else s3_num_idx
            addr_idx = s2_addr_idx if is_s2 else s3_addr_idx
            sx_idx = s2_sx_idx if is_s2 else s3_sx_idx

            # 1. Name tokens (>= 3 chars, non-stopword, capped at 3000)
            for tok in rec['name_tokens']:
                if len(tok) >= 3 and tok not in STOPWORDS and len(name_idx[tok]) < MAX_INDEX_LEN:
                    name_idx[tok].append(pool_idx)

            # 2. First 4-character prefix (capped at 3000)
            if len(rec['norm_name']) >= 4 and len(pref_idx[rec['norm_name'][:4]]) < MAX_INDEX_LEN:
                pref_idx[rec['norm_name'][:4]].append(pool_idx)

            # 3. Address numbers (>= 2 digits, capped at 3000)
            for num in rec['addr_numbers']:
                if len(num) >= 2 and len(num_idx[num]) < MAX_INDEX_LEN:
                    num_idx[num].append(pool_idx)

            # 4. Rare address tokens (>= 5 chars, non-stopword, capped at 3000)
            for tok in rec['norm_addr'].split():
                if len(tok) >= 5 and tok not in COMMON_ADDR_STOP and len(addr_idx[tok]) < MAX_INDEX_LEN:
                    addr_idx[tok].append(pool_idx)

            # 5. Phonetic Soundex tokens (for India transliterations)
            if is_india:
                for tok in rec['name_tokens']:
                    if len(tok) >= 3 and tok not in STOPWORDS:
                        sx = soundex(tok)
                        if sx and len(sx_idx[sx]) < MAX_INDEX_LEN:
                            sx_idx[sx].append(pool_idx)

            if (i + 1) % 1000000 == 0:
                print(f"    Indexed {i+1:,} pool records in {time.time()-t0:.1f}s...")

    print(f"  Indexed {len(pool_records):,} pool records in {time.time()-t0:.1f}s")
    s2_cnt = sum(1 for r in pool_records if r['entity_id'].startswith('S2-'))
    s3_cnt = len(pool_records) - s2_cnt
    print(f"  Independent Index Stats: S2={s2_cnt:,} records, S3={s3_cnt:,} records")
    print(f"  S2 tokens: {len(s2_name_idx):,}, prefixes: {len(s2_pref_idx):,}, nums: {len(s2_num_idx):,}")
    print(f"  S3 tokens: {len(s3_name_idx):,}, prefixes: {len(s3_pref_idx):,}, nums: {len(s3_num_idx):,}")

    return pool_records, (s2_name_idx, s2_pref_idx, s2_num_idx, s2_addr_idx, s2_sx_idx), \
                         (s3_name_idx, s3_pref_idx, s3_num_idx, s3_addr_idx, s3_sx_idx)


def query_source_candidates(s1_rec, indexes, is_india: bool, top_k: int = 15):
    """
    Retrieve Top-K candidates from an individual source's inverted index.
    Slices ultra-common postings at 1000 for sub-millisecond per-entity querying.
    """
    name_idx, pref_idx, num_idx, addr_idx, sx_idx = indexes
    sc = defaultdict(int)

    # 1. Name tokens
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok not in STOPWORDS:
            w = 8 if len(tok) >= 5 else 5
            postings = name_idx.get(tok, ())
            if len(postings) > 1000:
                postings = postings[:1000]
            for pi in postings:
                sc[pi] += w

    # 2. Phonetic Soundex (for India transliterations)
    if is_india:
        for tok in s1_rec['name_tokens']:
            if len(tok) >= 3 and tok not in STOPWORDS:
                sx = soundex(tok)
                if sx:
                    postings = sx_idx.get(sx, ())
                    if len(postings) > 800:
                        postings = postings[:800]
                    for pi in postings:
                        sc[pi] += 4

    # 3. 4-char Prefix
    if len(s1_rec['norm_name']) >= 4:
        postings = pref_idx.get(s1_rec['norm_name'][:4], ())
        if len(postings) > 1000:
            postings = postings[:1000]
        for pi in postings:
            sc[pi] += 4

    # 4. Address numbers (house numbers, unit numbers, PIN codes)
    for num in s1_rec['addr_numbers']:
        if len(num) >= 2:
            w = 12 if len(num) == 6 else 4
            postings = num_idx.get(num, ())
            if len(postings) > 1000:
                postings = postings[:1000]
            for pi in postings:
                sc[pi] += w

    # 5. Distinctive address tokens
    for tok in s1_rec['norm_addr'].split():
        if len(tok) >= 5 and tok not in COMMON_ADDR_STOP:
            matches = addr_idx.get(tok, ())
            if len(matches) < 350:
                for pi in matches:
                    sc[pi] += 5

    if not sc:
        return []
    return [pi for pi, _ in sorted(sc.items(), key=lambda x: -x[1])[:top_k]]


def predict_for_country(country: str, partition_dir: str, model, threshold: float,
                        rescue_threshold: float = 0.45,
                        max_candidates_per_source: int = 15,
                        chunk_size: int = 15000):
    """
    Run balanced inference for a single country and stream results to country checkpoint files.
    Includes singleton-rescuing logic and transitive graph triangle closure.
    """
    pool_path = os.path.join(partition_dir, "pool.tsv")
    s1_path = os.path.join(partition_dir, "s1.tsv")
    matching_ckpt = os.path.join(OUTPUT_DIR, f"matching_{country}.tsv")
    candidate_ckpt = os.path.join(OUTPUT_DIR, f"candidate_{country}.tsv")

    if not os.path.exists(pool_path) or not os.path.exists(s1_path):
        print(f"  ⚠️ Skipping {country}: files missing.")
        return 0, 0

    expected_s1 = EXPECTED_COUNTS.get(country)
    if os.path.exists(matching_ckpt) and os.path.exists(candidate_ckpt):
        with open(matching_ckpt, 'rb') as f:
            existing_lines = sum(1 for _ in f) - 1
        if expected_s1 and existing_lines == expected_s1:
            print(f"\n{'='*70}")
            print(f"  ⏭️ CHECKPOINT FOUND FOR {country} ({existing_lines:,} entities). SKIPPING TO NEXT.")
            print(f"{'='*70}")
            matched_cnt = 0
            with open(matching_ckpt, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) > 1 and p[1].strip():
                        matched_cnt += 1
            return existing_lines, matched_cnt

    print(f"\n{'='*70}")
    print(f"  PROCESSING COUNTRY: {country} (Match Threshold = {threshold:.2f}, Rescue = {rescue_threshold:.2f})")
    print(f"{'='*70}")
    t_start = time.time()

    is_india = (country == 'India')

    # Load and build dual independent indexes
    pool_records, s2_indexes, s3_indexes = build_country_dual_index(pool_path, country)

    matching_out_handle = open(matching_ckpt, 'w', encoding='utf-8')
    candidate_out_handle = open(candidate_ckpt, 'w', encoding='utf-8')
    matching_out_handle.write("source1_entity_id\tmatched_entity_ids\n")
    candidate_out_handle.write("source1_entity_id\tcandidate_entity_ids\n")

    # Stream S1 in chunks
    print(f"\n  [2/3] Streaming S1 and predicting matches...")
    total_s1 = 0
    total_matched_s1 = 0
    total_singletons = 0
    total_rescued = 0
    total_transitive = 0
    total_pairs_evaluated = 0

    chunk_s1 = []

    def process_chunk(s1_batch):
        nonlocal total_s1, total_matched_s1, total_singletons, total_rescued, total_transitive, total_pairs_evaluated
        if not s1_batch:
            return

        batch_pairs = []
        batch_pair_indices = []  # (s1_local_idx, pool_idx)
        s1_candidates_map = defaultdict(list)

        for s1_local_idx, s1_rec in enumerate(s1_batch):
            # Query S2 and S3 independently (25 from each, max 50 total)
            c2_pool_indices = query_source_candidates(s1_rec, s2_indexes, is_india, top_k=max_candidates_per_source)
            c3_pool_indices = query_source_candidates(s1_rec, s3_indexes, is_india, top_k=max_candidates_per_source)
            pool_indices = c2_pool_indices + c3_pool_indices

            if not pool_indices:
                continue

            s1_candidates_map[s1_local_idx] = pool_indices

            for pi in pool_indices:
                feats = compute_features(s1_rec, pool_records[pi])
                feat_vals = [feats[fn] for fn in FEATURE_NAMES]
                batch_pairs.append(feat_vals)
                batch_pair_indices.append((s1_local_idx, pi))

        # Model batch prediction
        if batch_pairs:
            X_batch = np.array(batch_pairs, dtype=np.float32)
            preds_proba = model.predict(X_batch)
            total_pairs_evaluated += len(batch_pairs)
        else:
            preds_proba = np.array([], dtype=np.float32)

        # Collect candidate probabilities per S1
        s1_candidate_probs = defaultdict(list)
        for (s1_local_idx, pi), prob in zip(batch_pair_indices, preds_proba):
            s1_candidate_probs[s1_local_idx].append((prob, pi))

        # Make decisions per S1 entity
        for s1_local_idx, s1_rec in enumerate(s1_batch):
            s1_id = s1_rec['entity_id']
            cand_pool_indices = s1_candidates_map.get(s1_local_idx, [])
            cand_ids = sorted(list({pool_records[pi]['entity_id'] for pi in cand_pool_indices}))

            cand_probs = s1_candidate_probs.get(s1_local_idx, [])

            # 1. Standard high-confidence threshold
            matched_pis = [pi for prob, pi in cand_probs if prob >= threshold]

            # 2. Singleton rescue: if no candidate met high threshold, rescue if top candidate is strong
            if not matched_pis and cand_probs:
                best_prob, best_pi = max(cand_probs, key=lambda x: x[0])
                if best_prob >= rescue_threshold:
                    matched_pis = [best_pi]
                    total_rescued += 1

            # 3. Transitive Graph Triangle Closure:
            # If S1 matched S2 with high confidence, close triangle to identical S3 candidate
            if matched_pis:
                s2_matched = [pi for pi in matched_pis if pool_records[pi]['entity_id'].startswith('S2-')]
                s3_cands = [pi for pi in cand_pool_indices if pool_records[pi]['entity_id'].startswith('S3-') and pi not in matched_pis]
                for s2_pi in s2_matched:
                    s2_rec = pool_records[s2_pi]
                    for s3_pi in s3_cands:
                        s3_rec = pool_records[s3_pi]
                        if s2_rec['norm_name'] == s3_rec['norm_name'] and s2_rec['norm_addr'] == s3_rec['norm_addr']:
                            matched_pis.append(s3_pi)
                            total_transitive += 1

            matched_ids = sorted(list({pool_records[pi]['entity_id'] for pi in matched_pis}))

            # Write chunk output
            matching_out_handle.write(f"{s1_id}\t{','.join(matched_ids)}\n")
            candidate_out_handle.write(f"{s1_id}\t{','.join(cand_ids)}\n")

            total_s1 += 1
            if matched_ids:
                total_matched_s1 += 1
            else:
                total_singletons += 1

        matching_out_handle.flush()
        candidate_out_handle.flush()

    with open(s1_path, 'r', encoding='utf-8') as f:
        header = next(f)
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3])
            chunk_s1.append(rec)

            if len(chunk_s1) >= chunk_size:
                process_chunk(chunk_s1)
                chunk_s1 = []
                elapsed = time.time() - t_start
                print(f"    Processed {total_s1:,} S1 entities ({total_matched_s1:,} matched, "
                      f"{total_singletons:,} singletons [{total_singletons/max(1,total_s1)*100:.1f}%], "
                      f"{total_rescued:,} rescued, {total_pairs_evaluated:,} pairs) — "
                      f"{total_s1/elapsed:.0f} S1/sec")

        if chunk_s1:
            process_chunk(chunk_s1)

    matching_out_handle.close()
    candidate_out_handle.close()

    elapsed = time.time() - t_start
    print(f"\n  ✅ Country {country} complete in {elapsed/60:.1f} minutes!")
    print(f"    Total S1: {total_s1:,} | Matched: {total_matched_s1:,} ({total_matched_s1/total_s1*100:.1f}%) | "
          f"Singletons: {total_singletons:,} ({total_singletons/total_s1*100:.1f}%)")
    print(f"    Rescued from singleton: {total_rescued:,} | Transitive closures: {total_transitive:,}")
    print(f"    Total pairs featurized & evaluated: {total_pairs_evaluated:,}")

    # Explicitly free memory
    del pool_records, s2_indexes, s3_indexes
    import gc
    gc.collect()

    return total_s1, total_matched_s1


def main():
    total_start = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — CHAMPIONSHIP PREDICTION PIPELINE")
    print("=" * 70)

    # Load trained models
    model_path = os.path.join(OUTPUT_DIR, 'models_ensemble.pkl')
    print(f"Loading trained ensemble models from {model_path}...")
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    models = saved['models']
    print(f"Loaded models for: {list(models.keys())}")

    # Optimal calibrated thresholds
    calibrated_thresholds = {
        'France': 0.75,
        'US': 0.78,
        'India': 0.75
    }
    rescue_thresholds = {
        'France': 0.45,
        'US': 0.45,
        'India': 0.45
    }
    print(f"Calibrated Thresholds: {calibrated_thresholds}")
    print(f"Rescue Thresholds: {rescue_thresholds}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    partitions_dir = "partitions/test"
    countries = ['France', 'US', 'India']

    grand_total_s1 = 0
    grand_total_matched = 0

    for country in countries:
        cdir = os.path.join(partitions_dir, country)
        if not os.path.exists(cdir):
            continue

        model = models[country] if country in models else models['US']
        thresh = calibrated_thresholds.get(country, 0.75)
        rescue_th = rescue_thresholds.get(country, 0.45)

        cnt_s1, cnt_matched = predict_for_country(
            country,
            cdir,
            model,
            thresh,
            rescue_threshold=rescue_th,
            max_candidates_per_source=15,
            chunk_size=15000,
        )

        grand_total_s1 += cnt_s1
        grand_total_matched += cnt_matched

    print(f"\n[3/3] Merging country checkpoints and aligning to test_source1.tsv order...")
    t0 = time.time()

    # Load all country predictions into memory
    preds_map = {}
    cands_map = {}

    for country in countries:
        matching_ckpt = os.path.join(OUTPUT_DIR, f"matching_{country}.tsv")
        candidate_ckpt = os.path.join(OUTPUT_DIR, f"candidate_{country}.tsv")

        if os.path.exists(matching_ckpt):
            print(f"  Loading predictions from {matching_ckpt}...")
            with open(matching_ckpt, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    preds_map[parts[0]] = parts[1] if len(parts) > 1 else ''

        if os.path.exists(candidate_ckpt):
            print(f"  Loading candidates from {candidate_ckpt}...")
            with open(candidate_ckpt, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    cands_map[parts[0]] = parts[1] if len(parts) > 1 else ''

    print(f"  Total mapped entities: {len(preds_map):,}. Writing strictly in test_source1.tsv order...")

    # Write in EXACT order of test_source1.tsv
    with open(MATCHING_OUTPUT, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        with open(TEST_S1, 'r', encoding='utf-8') as f_in:
            next(f_in)
            for line in f_in:
                eid = line.strip().split('\t')[0]
                m = preds_map.get(eid, '')
                f_out.write(f"{eid}\t{m}\n")

    with open(CANDIDATE_OUTPUT, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        with open(TEST_S1, 'r', encoding='utf-8') as f_in:
            next(f_in)
            for line in f_in:
                eid = line.strip().split('\t')[0]
                c = cands_map.get(eid, '')
                f_out.write(f"{eid}\t{c}\n")

    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  🏆 PREDICTION COMPLETE!")
    print(f"  Total test entities: {len(preds_map):,}")
    matched_count = sum(1 for v in preds_map.values() if v)
    print(f"  Predicted matched   : {matched_count:,} ({matched_count/len(preds_map)*100:.1f}%)")
    print(f"  Predicted singletons: {len(preds_map) - matched_count:,} ({(len(preds_map) - matched_count)/len(preds_map)*100:.1f}%)")
    print(f"  Files created:")
    print(f"    - {MATCHING_OUTPUT} ({os.path.getsize(MATCHING_OUTPUT)/1e6:.1f} MB)")
    print(f"    - {CANDIDATE_OUTPUT} ({os.path.getsize(CANDIDATE_OUTPUT)/1e6:.1f} MB)")
    print(f"  Total pipeline time : {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # Run verification audit
    from src.verify_submission import verify
    verify()


if __name__ == '__main__':
    main()
