import os
import sys
import time
import math
from collections import defaultdict
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.config import *
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.train_ensemble import EnsemblePredictor
import __main__
__main__.EnsemblePredictor = EnsemblePredictor
import pickle

print("=" * 70)
print("TESTING NEW HIGH-RECALL INDEX ON INDIA TEST DATA")
print("=" * 70)

# Load ensemble model
with open('output/models_ensemble.pkl', 'rb') as f:
    saved = pickle.load(f)
model = saved['models']['India']
print("Loaded India ensemble model.")

STOPWORDS = {
    'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd',
    'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions',
    'enterprises', 'international', 'holdings', 'global', 'technologies',
    'sarl', 'sasu', 'eurl', 'association'
}

def soundex(name: str) -> str:
    if not name or not name.isalpha(): return ''
    name = name.upper()
    codes = {'B':'1','F':'1','P':'1','V':'1','C':'2','G':'2','J':'2','K':'2','Q':'2','S':'2','X':'2','Z':'2','D':'3','T':'3','L':'4','M':'5','N':'5','R':'6'}
    first = name[0]
    tail = [codes.get(c, '') for c in name[1:]]
    res = [first]
    prev = codes.get(first, '')
    for c in tail:
        if c != prev and c != '': res.append(c)
        prev = c
    return (''.join(res) + '0000')[:4]

# 1. Index full India pool
pool_path = 'partitions/test/India/pool.tsv'
t0 = time.time()
pool_records = []

s2_name_idx = defaultdict(list)
s2_pref_idx = defaultdict(list)
s2_num_idx = defaultdict(list)
s2_sx_idx = defaultdict(list)

s3_name_idx = defaultdict(list)
s3_pref_idx = defaultdict(list)
s3_num_idx = defaultdict(list)
s3_sx_idx = defaultdict(list)

print(f"Indexing full India pool from {pool_path}...")
with open(pool_path, 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) < 4: continue
        rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', 'India')
        pi = len(pool_records)
        pool_records.append(rec)

        is_s2 = parts[0].startswith('S2-')
        name_idx = s2_name_idx if is_s2 else s3_name_idx
        pref_idx = s2_pref_idx if is_s2 else s3_pref_idx
        num_idx = s2_num_idx if is_s2 else s3_num_idx
        sx_idx = s2_sx_idx if is_s2 else s3_sx_idx

        # 1. Name tokens - DO NOT CAP AT 600!
        for tok in rec['name_tokens']:
            if len(tok) >= 3 and tok not in STOPWORDS:
                name_idx[tok].append(pi)
                sx = soundex(tok)
                if sx:
                    sx_idx[sx].append(pi)

        # 2. 4-char prefix
        if len(rec['norm_name']) >= 4:
            pref_idx[rec['norm_name'][:4]].append(pi)

        # 3. Address numbers
        for num in rec['addr_numbers']:
            if len(num) >= 2:
                num_idx[num].append(pi)

        if (i + 1) % 1000000 == 0:
            print(f"  Indexed {i+1:,} records in {time.time()-t0:.1f}s...")

print(f"Total pool records indexed: {len(pool_records):,} in {time.time()-t0:.1f}s")

# Compute IDFs
N_pool = len(pool_records)
def compute_idfs(idx):
    return {k: math.log(N_pool / (len(v) + 1.0)) for k, v in idx.items()}

s2_name_idf = compute_idfs(s2_name_idx)
s3_name_idf = compute_idfs(s3_name_idx)
s2_num_idf = compute_idfs(s2_num_idx)
s3_num_idf = compute_idfs(s3_num_idx)

def query_candidates(s1_rec, indexes, idf_maps, top_k=25, max_postings=5000):
    name_idx, pref_idx, num_idx, sx_idx = indexes
    name_idf, num_idf = idf_maps
    sc = defaultdict(float)

    # Name tokens
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok in name_idx:
            postings = name_idx[tok]
            if len(postings) <= max_postings:
                idf = name_idf.get(tok, 1.0)
                w = idf * 2.5
                for pi in postings:
                    sc[pi] += w

    # Prefix
    if len(s1_rec['norm_name']) >= 4:
        pref = s1_rec['norm_name'][:4]
        if pref in pref_idx:
            postings = pref_idx[pref]
            if len(postings) <= max_postings:
                for pi in postings:
                    sc[pi] += 3.0

    # Address numbers (PIN codes, house numbers)
    for num in s1_rec['addr_numbers']:
        if len(num) >= 2 and num in num_idx:
            postings = num_idx[num]
            if len(postings) <= max_postings:
                idf = num_idf.get(num, 1.0)
                w = idf * 3.0 if len(num) == 6 else idf * 1.0
                for pi in postings:
                    sc[pi] += w

    # Soundex
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok not in STOPWORDS:
            sx = soundex(tok)
            if sx and sx in sx_idx:
                postings = sx_idx[sx]
                if len(postings) <= 1500:
                    for pi in postings:
                        sc[pi] += 1.5

    if not sc:
        return []
    if len(sc) <= top_k:
        return list(sc.keys())
    return [pi for pi, _ in sorted(sc.items(), key=lambda x: -x[1])[:top_k]]

s2_idxs = (s2_name_idx, s2_pref_idx, s2_num_idx, s2_sx_idx)
s3_idxs = (s3_name_idx, s3_pref_idx, s3_num_idx, s3_sx_idx)
s2_idfs = (s2_name_idf, s2_num_idf)
s3_idfs = (s3_name_idf, s3_num_idf)

# Now test on 1,000 test S1 entities
s1_test = []
with open('partitions/test/India/s1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        s1_test.append(normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', 'India'))
        if len(s1_test) >= 1000:
            break

print(f"\nEvaluating on {len(s1_test):,} India test entities...")
t_eval = time.time()
max_probs = []
cands_per_entity = []

for s1_rec in s1_test:
    c2 = query_candidates(s1_rec, s2_idxs, s2_idfs, top_k=25)
    c3 = query_candidates(s1_rec, s3_idxs, s3_idfs, top_k=25)
    pool_indices = c2 + c3
    cands_per_entity.append(len(pool_indices))

    if not pool_indices:
        max_probs.append(0.0)
        continue

    pair_feats = []
    for pi in pool_indices:
        feats = compute_features(s1_rec, pool_records[pi])
        pair_feats.append([feats[fn] for fn in FEATURE_NAMES])

    probs = model.predict(np.array(pair_feats, dtype=np.float32))
    max_probs.append(float(np.max(probs)))

max_probs = np.array(max_probs)
elapsed = time.time() - t_eval
print(f"Evaluated {len(s1_test)} entities in {elapsed:.1f}s ({len(s1_test)/elapsed:.0f} S1/sec)")
print(f"Avg candidates per entity: {np.mean(cands_per_entity):.1f}")
print(f"Zero candidate entities: {np.sum(np.array(cands_per_entity) == 0)}/{len(s1_test)}")

print("\n--- Max Probability Percentiles with New Index ---")
for p in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
    print(f"  {p:2d}th percentile: {np.percentile(max_probs, p):.4f}")

print("\n--- Empty / Singleton Rate at Various Thresholds ---")
for th in [0.80, 0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30]:
    rate = np.mean(max_probs < th) * 100
    print(f"  Threshold {th:.2f}: {rate:.1f}% singletons (Target: ~5.6%)")
