import sys, os, time, pickle, random, math
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, '.')
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.evaluate import compute_macro_f05, compute_f05_per_entity
from src.fast_predict import soundex

# Load ensemble models - extract LightGBM model
with open('output/models_ensemble.pkl', 'rb') as f:
    saved = pickle.load(f)
# In EnsemblePredictor, lgb_model is the LightGBM Booster
lgb_india = saved['models']['India'].lgb_model
lgb_us = saved['models']['US'].lgb_model

# Load GT
gt = {}
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        gt[p[0]] = set(p[1].split(',')) if len(p) > 1 and p[1].strip() else set()

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}
COMMON_ADDR = {'road', 'street', 'lane', 'floor', 'building', 'near', 'opposite', 'behind', 'india', 'state', 'city', 'cross', 'main'}

# Sample 1,000 India S1 entities
random.seed(42)
s1_test = []
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 3 and p[3] == 'India':
            s1_test.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3]))
            if len(s1_test) >= 1000:
                break

needed_matches = set()
for r in s1_test:
    needed_matches.update(gt.get(r['entity_id'], set()))

# Load S2 and S3 pool (needed matches + 300,000 real pool records)
s2_records = []
s3_records = []

for path in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    is_s2 = 'source2' in path
    target_list = s2_records if is_s2 else s3_records
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) > 3 and p[3] == 'India':
                if p[0] in needed_matches or (len(target_list) < 150000 and random.random() < 0.2):
                    target_list.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3]))

def build_lightning_index(pool, max_postings=400):
    N = len(pool)
    name_df = Counter()
    addr_df = Counter()
    num_df = Counter()
    sx_df = Counter()
    
    for rec in pool:
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
        
    name_idx = defaultdict(list)
    num_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    sx_idx = defaultdict(list)
    
    for idx, rec in enumerate(pool):
        for t in rec['name_tokens']:
            if len(t) >= 3 and t not in STOPWORDS and name_df[t] < 0.01 * N:
                if len(name_idx[t]) < max_postings:
                    name_idx[t].append(idx)
            sx = soundex(t)
            if sx and sx_df[sx] < 0.01 * N:
                if len(sx_idx[sx]) < max_postings:
                    sx_idx[sx].append(idx)
        for num in rec['addr_numbers']:
            if num_df[num] < 0.01 * N:
                if len(num_idx[num]) < max_postings:
                    num_idx[num].append(idx)
        for t in rec['norm_addr'].split():
            if len(t) >= 3 and t not in COMMON_ADDR and addr_df[t] < 0.01 * N:
                if len(addr_idx[t]) < max_postings:
                    addr_idx[t].append(idx)
                
    return (name_idx, num_idx, addr_idx, sx_idx), (name_df, num_df, addr_df, sx_df), idf

s2_idx, s2_df, s2_idf = build_lightning_index(s2_records)
s3_idx, s3_df, s3_idf = build_lightning_index(s3_records)

def query_lightning(s1_rec, indexes, dfs, idf_fn, top_k=12):
    name_idx, num_idx, addr_idx, sx_idx = indexes
    name_df, num_df, addr_df, sx_df = dfs
    scores = defaultdict(float)
    
    for t in s1_rec['name_tokens']:
        if t in name_idx:
            sc = idf_fn(name_df[t]) * 3.0
            for pi in name_idx[t]:
                scores[pi] += sc
        sx = soundex(t)
        if sx in sx_idx:
            sc = idf_fn(sx_df[sx]) * 1.5
            for pi in sx_idx[sx]:
                scores[pi] += sc
                
    for num in s1_rec['addr_numbers']:
        if num in num_idx:
            w = 4.0 if len(num) == 6 else 1.5
            sc = idf_fn(num_df[num]) * w
            for pi in num_idx[num]:
                scores[pi] += sc
                
    for t in s1_rec['norm_addr'].split():
        if t in addr_idx:
            sc = idf_fn(addr_df[t]) * 1.5
            for pi in addr_idx[t]:
                scores[pi] += sc
                
    if not scores:
        return []
    # Only keep candidates with score > 2.0 (filters out zero-overlap noise!)
    filtered = [(pi, s) for pi, s in scores.items() if s >= 2.0]
    filtered.sort(key=lambda x: -x[1])
    return [pi for pi, _ in filtered[:top_k]]

t0 = time.time()
preds = {}
cand_hits = 0
total_true = sum(len(gt.get(r['entity_id'], set())) for r in s1_test)
total_pairs = 0

for s1_rec in s1_test:
    eid = s1_rec['entity_id']
    true_set = gt.get(eid, set())
    
    s2_cands = query_lightning(s1_rec, s2_idx, s2_df, s2_idf, top_k=12)
    s3_cands = query_lightning(s1_rec, s3_idx, s3_df, s3_idf, top_k=12)
    
    cand_ids = [s2_records[pi]['entity_id'] for pi in s2_cands] + [s3_records[pi]['entity_id'] for pi in s3_cands]
    cand_hits += len(set(cand_ids) & true_set)
    
    pairs = []
    cids = []
    s2_recs_map = {}
    s3_recs_map = {}
    
    for pi in s2_cands:
        rec = s2_records[pi]
        feats = compute_features(s1_rec, rec)
        pairs.append([feats[fn] for fn in FEATURE_NAMES])
        cids.append(rec['entity_id'])
        s2_recs_map[rec['entity_id']] = rec
        
    for pi in s3_cands:
        rec = s3_records[pi]
        feats = compute_features(s1_rec, rec)
        pairs.append([feats[fn] for fn in FEATURE_NAMES])
        cids.append(rec['entity_id'])
        s3_recs_map[rec['entity_id']] = rec
        
    total_pairs += len(pairs)
    matched = set()
    if pairs:
        X = np.array(pairs, dtype=np.float32)
        probs = lgb_india.predict(X)
        for i, cid in enumerate(cids):
            if probs[i] >= 0.80:
                matched.add(cid)
                
    # Transitive triangle closure
    s2_matched = [s2_recs_map[cid] for cid in matched if cid.startswith('S2-')]
    for s2_rec in s2_matched:
        for s3_cid, s3_rec in s3_recs_map.items():
            if s3_cid not in matched:
                if s2_rec['norm_name'] == s3_rec['norm_name'] and (not s2_rec['norm_addr'] or not s3_rec['norm_addr'] or s2_rec['norm_addr'] == s3_rec['norm_addr']):
                    matched.add(s3_cid)
                    
    preds[eid] = matched

elapsed = time.time() - t0
print(f"Processed {len(s1_test)} entities ({total_pairs} pairs) in {elapsed:.2f}s ({len(s1_test)/elapsed:.0f} S1/sec)")
print(f"Candidate Blocking Recall: {cand_hits} / {total_true} ({cand_hits/total_true*100:.2f}%)")

scores = [compute_f05_per_entity(preds[r['entity_id']], gt.get(r['entity_id'], set())) for r in s1_test]
macro_f05 = sum(scores) / len(scores)
perfect = sum(1 for s in scores if s >= 0.99)
zeros = sum(1 for s in scores if s == 0)

print(f"\n=======================================================")
print(f"  LIGHTNING ENGINE MACRO F0.5 SCORE: {macro_f05:.6f}")
print(f"  Perfect Matches (>= 0.99)        : {perfect} / {len(s1_test)} ({perfect/len(s1_test)*100:.1f}%)")
print(f"  Zeros                            : {zeros} / {len(s1_test)} ({zeros/len(s1_test)*100:.1f}%)")
print(f"=======================================================")
