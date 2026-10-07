import os
import sys
import time
import pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from src.config import *
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.train_ensemble import EnsemblePredictor
from src.fast_predict import soundex, STOPWORDS, COMMON_ADDR_STOP, query_source_candidates
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

print("Profiling fast_predict on France slice...")

# Load models
with open('output/models_ensemble.pkl', 'rb') as f:
    saved = pickle.load(f)
model = saved['models']['US']

# Load 500 S1 records
s1_recs = []
with open('partitions/test/France/s1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        s1_recs.append(normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', 'France'))
        if len(s1_recs) >= 500:
            break

# Load 100,000 pool records
pool_records = []
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

MAX_INDEX_LEN = 3000

t0 = time.time()
with open('partitions/test/France/pool.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) < 4: continue
        rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', 'France')
        pi = len(pool_records)
        pool_records.append(rec)

        is_s2 = parts[0].startswith('S2-')
        name_idx = s2_name_idx if is_s2 else s3_name_idx
        pref_idx = s2_pref_idx if is_s2 else s3_pref_idx
        num_idx = s2_num_idx if is_s2 else s3_num_idx
        addr_idx = s2_addr_idx if is_s2 else s3_addr_idx

        for tok in rec['name_tokens']:
            if len(tok) >= 3 and tok not in STOPWORDS and len(name_idx[tok]) < MAX_INDEX_LEN:
                name_idx[tok].append(pi)

        if len(rec['norm_name']) >= 4 and len(pref_idx[rec['norm_name'][:4]]) < MAX_INDEX_LEN:
            pref_idx[rec['norm_name'][:4]].append(pi)

        for num in rec['addr_numbers']:
            if len(num) >= 2 and len(num_idx[num]) < MAX_INDEX_LEN:
                num_idx[num].append(pi)

        for tok in rec['norm_addr'].split():
            if len(tok) >= 5 and tok not in COMMON_ADDR_STOP and len(addr_idx[tok]) < MAX_INDEX_LEN:
                addr_idx[tok].append(pi)

        if len(pool_records) >= 100000:
            break

print(f"Indexed 100k pool records in {time.time()-t0:.2f}s")

s2_indexes = (s2_name_idx, s2_pref_idx, s2_num_idx, s2_addr_idx, s2_sx_idx)
s3_indexes = (s3_name_idx, s3_pref_idx, s3_num_idx, s3_addr_idx, s3_sx_idx)

# Step 1: Time candidate querying
t1 = time.time()
all_candidates = []
for s1_rec in s1_recs:
    c2 = query_source_candidates(s1_rec, s2_indexes, False, top_k=25)
    c3 = query_source_candidates(s1_rec, s3_indexes, False, top_k=25)
    all_candidates.append(c2 + c3)
t_query = time.time() - t1
total_cands = sum(len(c) for c in all_candidates)
print(f"Candidate querying: {t_query:.2f}s for 500 entities ({t_query/500*1000:.2f} ms/entity). Total candidates: {total_cands}")

# Step 2: Time feature extraction
t2 = time.time()
batch_pairs = []
for s1_rec, cands in zip(s1_recs, all_candidates):
    for pi in cands:
        feats = compute_features(s1_rec, pool_records[pi])
        batch_pairs.append([feats[fn] for fn in FEATURE_NAMES])
t_feat = time.time() - t2
print(f"Feature computation: {t_feat:.2f}s for {len(batch_pairs)} pairs ({t_feat/len(batch_pairs)*1000:.3f} ms/pair)")

# Step 3: Time model prediction
t3 = time.time()
if batch_pairs:
    X = np.array(batch_pairs, dtype=np.float32)
    preds = model.predict(X)
t_pred = time.time() - t3
print(f"Model prediction: {t_pred:.2f}s for {len(batch_pairs)} pairs")

print("\n--- Summary Breakdown for 500 entities ---")
print(f"  Candidate query: {t_query:.2f}s ({t_query/(t_query+t_feat+t_pred)*100:.1f}%)")
print(f"  Featurization  : {t_feat:.2f}s ({t_feat/(t_query+t_feat+t_pred)*100:.1f}%)")
print(f"  Model predict  : {t_pred:.2f}s ({t_pred/(t_query+t_feat+t_pred)*100:.1f}%)")
