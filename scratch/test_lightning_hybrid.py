import os, sys, time, pickle, math
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, '.')
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.fast_predict import soundex
from src.train_ensemble import EnsemblePredictor
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

# Load ensemble models
with open('output/models_ensemble.pkl', 'rb') as f:
    saved = pickle.load(f)
ens_us = saved['models']['US']

# Sample 1000 S1 from US
s1_test = []
with open('partitions/test/US/s1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        s1_test.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', 'US'))
        if len(s1_test) >= 1000:
            break

# Load 50,000 pool records
pool_recs = []
with open('partitions/test/US/pool.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) < 4: continue
        pool_recs.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', 'US'))
        if len(pool_recs) >= 50000:
            break

s2_recs = [r for r in pool_recs if r['entity_id'].startswith('S2-')]
s3_recs = [r for r in pool_recs if r['entity_id'].startswith('S3-')]

print(f"Loaded {len(s1_test)} S1, {len(s2_recs)} S2, {len(s3_recs)} S3 records")

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}
COMMON_ADDR = {'road', 'street', 'lane', 'floor', 'building', 'near', 'opposite', 'behind', 'beside', 'plot', 'shop', 'flat', 'apartment', 'nagar', 'colony', 'india', 'state', 'city', 'cross', 'main', 'sector'}

def build_fast_idx(records):
    N = len(records)
    name_df, addr_df, num_df, sx_df = Counter(), Counter(), Counter(), Counter()
    for rec in records:
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
    def idf(df):
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)
    name_idx, num_idx, addr_idx, sx_idx = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for idx, rec in enumerate(records):
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
    return (name_idx, num_idx, addr_idx, sx_idx), (name_df, num_df, addr_df, sx_df), idf

s2_idx, s2_df, s2_idf = build_fast_idx(s2_recs)
s3_idx, s3_df, s3_idf = build_fast_idx(s3_recs)

def query_fast(s1_rec, indexes, dfs, idf_fn, top_k=15):
    name_idx, num_idx, addr_idx, sx_idx = indexes
    name_df, num_df, addr_df, sx_df = dfs
    scores = defaultdict(float)
    for t in s1_rec['name_tokens']:
        if t in name_idx:
            sc = idf_fn(name_df[t]) * 3.0
            for pi in name_idx[t][:1000]:
                scores[pi] += sc
        sx = soundex(t)
        if sx in sx_idx:
            sc = idf_fn(sx_df[sx]) * 1.5
            for pi in sx_idx[sx][:500]:
                scores[pi] += sc
    for num in s1_rec['addr_numbers']:
        if num in num_idx:
            w = 4.0 if len(num) == 6 else 1.5
            sc = idf_fn(num_df[num]) * w
            for pi in num_idx[num][:1000]:
                scores[pi] += sc
    for t in s1_rec['norm_addr'].split():
        if t in addr_idx:
            sc = idf_fn(addr_df[t]) * 1.5
            for pi in addr_idx[t][:1000]:
                scores[pi] += sc
    if not scores:
        return []
    filtered = [(pi, s) for pi, s in scores.items() if s >= 1.5]
    filtered.sort(key=lambda x: -x[1])
    return [pi for pi, _ in filtered[:top_k]]

t0 = time.time()
batch_pairs = []
batch_meta = []
for s1_idx, s1_rec in enumerate(s1_test):
    c2 = query_fast(s1_rec, s2_idx, s2_df, s2_idf, top_k=15)
    c3 = query_fast(s1_rec, s3_idx, s3_df, s3_idf, top_k=15)
    for pi in c2:
        rec = s2_recs[pi]
        feats = compute_features(s1_rec, rec)
        batch_pairs.append([feats[fn] for fn in FEATURE_NAMES])
        batch_meta.append((s1_idx, rec['entity_id'], rec))
    for pi in c3:
        rec = s3_recs[pi]
        feats = compute_features(s1_rec, rec)
        batch_pairs.append([feats[fn] for fn in FEATURE_NAMES])
        batch_meta.append((s1_idx, rec['entity_id'], rec))

t_feat = time.time() - t0
print(f"Candidate query + Featurization: {t_feat:.2f}s for {len(s1_test)} entities ({len(batch_pairs)} pairs)")

t1 = time.time()
if batch_pairs:
    X = np.array(batch_pairs, dtype=np.float32)
    # Stage 1: Fast LightGBM predict
    lgb_probs = ens_us.lgb_model.predict(X)
    
    # Stage 2: Only evaluate XGB + Cat for pairs where lgb_probs >= 0.20
    candidate_mask = lgb_probs >= 0.20
    final_probs = lgb_probs.copy()
    if np.any(candidate_mask):
        sub_X = X[candidate_mask]
        xgb_sub = ens_us.xgb_model.predict_proba(sub_X)[:, 1]
        cat_sub = ens_us.cb_model.predict_proba(sub_X)[:, 1]
        final_probs[candidate_mask] = (lgb_probs[candidate_mask] + xgb_sub + cat_sub) / 3.0
t_pred = time.time() - t1
print(f"Gated Tri-Ensemble predict: {t_pred:.2f}s (candidates above 0.20: {np.sum(candidate_mask)} / {len(batch_pairs)})")

total_t = t_feat + t_pred
print(f"Total time for 1,000 entities: {total_t:.2f}s ({len(s1_test)/total_t:.0f} S1/sec)!")
