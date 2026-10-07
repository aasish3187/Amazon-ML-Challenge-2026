import sys, os, time, pickle, random, math
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, '.')
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.evaluate import compute_macro_f05, compute_f05_per_entity
from src.fast_predict import soundex
from src.train_ensemble import EnsemblePredictor
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

# Load ensemble models
with open('output/models_ensemble.pkl', 'rb') as f:
    saved = pickle.load(f)
models = saved['models']

# Load GT
gt = {}
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        gt[p[0]] = set(p[1].split(',')) if len(p) > 1 and p[1].strip() else set()

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}
COMMON_ADDR = {'road', 'street', 'lane', 'floor', 'building', 'near', 'opposite', 'behind', 'india', 'state', 'city', 'cross', 'main'}

# Test on India sample (1,000 entities)
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

# Load S2 and S3 pool
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
                if p[0] in needed_matches or (len(target_list) < 200000 and random.random() < 0.25):
                    target_list.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3]))

def build_idf_index(pool):
    N = len(pool)
    name_df = Counter()
    addr_df = Counter()
    num_df = Counter()
    sx_df = Counter()
    pref_df = Counter()
    
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
        if len(rec['norm_name']) >= 4:
            pref_df[rec['norm_name'][:4]] += 1
                
    def idf(df):
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
    name_idx = defaultdict(list)
    num_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    sx_idx = defaultdict(list)
    pref_idx = defaultdict(list)
    
    for idx, rec in enumerate(pool):
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

s2_idx, s2_df, s2_idf = build_idf_index(s2_records)
s3_idx, s3_df, s3_idf = build_idf_index(s3_records)

def query_top_k(s1_rec, indexes, dfs, idf_fn, pool, top_k=30):
    name_idx, num_idx, addr_idx, sx_idx, pref_idx = indexes
    name_df, num_df, addr_df, sx_df, pref_df = dfs
    scores = defaultdict(float)
    
    for t in s1_rec['name_tokens']:
        if t in name_idx:
            sc = idf_fn(name_df[t]) * 3.0
            for pi in name_idx[t][:1500]:
                scores[pi] += sc
        sx = soundex(t)
        if sx in sx_idx:
            sc = idf_fn(sx_df[sx]) * 1.5
            for pi in sx_idx[sx][:800]:
                scores[pi] += sc
                
    for num in s1_rec['addr_numbers']:
        if num in num_idx:
            w = 4.0 if len(num) == 6 else 1.5
            sc = idf_fn(num_df[num]) * w
            for pi in num_idx[num][:1500]:
                scores[pi] += sc
                
    for t in s1_rec['norm_addr'].split():
        if t in addr_idx:
            sc = idf_fn(addr_df[t]) * 1.5
            for pi in addr_idx[t][:1500]:
                scores[pi] += sc
                
    if len(s1_rec['norm_name']) >= 4:
        pref = s1_rec['norm_name'][:4]
        if pref in pref_idx:
            sc = idf_fn(pref_df[pref]) * 1.2
            for pi in pref_idx[pref][:800]:
                scores[pi] += sc
                
    if not scores:
        return []
    return [pi for pi, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]

model = models['India']
total_true_matches = sum(len(gt.get(r['entity_id'], set())) for r in s1_test)
cand_hits = 0

entity_cand_probs = []

for s1_rec in s1_test:
    eid = s1_rec['entity_id']
    true_set = gt.get(eid, set())
    
    s2_cands = query_top_k(s1_rec, s2_idx, s2_df, s2_idf, s2_records, top_k=30)
    s3_cands = query_top_k(s1_rec, s3_idx, s3_df, s3_idf, s3_records, top_k=30)
    
    cand_ids = [s2_records[pi]['entity_id'] for pi in s2_cands] + [s3_records[pi]['entity_id'] for pi in s3_cands]
    cand_hits += len(set(cand_ids) & true_set)
    
    pairs = []
    pair_cids = []
    pair_recs = []
    for pi in s2_cands:
        feats = compute_features(s1_rec, s2_records[pi])
        pairs.append([feats[fn] for fn in FEATURE_NAMES])
        pair_cids.append(s2_records[pi]['entity_id'])
        pair_recs.append(s2_records[pi])
    for pi in s3_cands:
        feats = compute_features(s1_rec, s3_records[pi])
        pairs.append([feats[fn] for fn in FEATURE_NAMES])
        pair_cids.append(s3_records[pi]['entity_id'])
        pair_recs.append(s3_records[pi])
        
    if pairs:
        X = np.array(pairs, dtype=np.float32)
        probs = model.predict(X)
        entity_cand_probs.append((s1_rec, true_set, list(zip(pair_cids, probs, pair_recs))))
    else:
        entity_cand_probs.append((s1_rec, true_set, []))

print(f"Candidate Blocking Recall (top_k=30 + prefix): {cand_hits} / {total_true_matches} ({cand_hits/total_true_matches*100:.2f}%)")

for thresh in [0.70, 0.75, 0.80, 0.85]:
    preds = {}
    for s1_rec, true_set, c_list in entity_cand_probs:
        eid = s1_rec['entity_id']
        matched = {cid for cid, prob, _ in c_list if prob >= thresh}
        
        # Transitive triangle closure: if S2 matched with high conf (>=0.85), check if any S3 cand matches S2 record
        s2_high = [rec for cid, prob, rec in c_list if prob >= 0.85 and cid.startswith('S2-')]
        s3_cands = [(cid, rec) for cid, prob, rec in c_list if cid.startswith('S3-') and cid not in matched]
        for s2_rec in s2_high:
            for s3_cid, s3_rec in s3_cands:
                if s2_rec['norm_name'] == s3_rec['norm_name'] and (not s2_rec['norm_addr'] or not s3_rec['norm_addr'] or s2_rec['norm_addr'] == s3_rec['norm_addr']):
                    matched.add(s3_cid)
                    
        preds[eid] = matched
        
    scores = [compute_f05_per_entity(preds[r['entity_id']], gt.get(r['entity_id'], set())) for r in s1_test]
    macro_f05 = sum(scores) / len(scores)
    perfect = sum(1 for s in scores if s >= 0.99)
    zeros = sum(1 for s in scores if s == 0)
    print(f"Threshold {thresh:.2f} + Transitive: Macro F0.5 = {macro_f05:.6f} | Perfect: {perfect}/1000 | Zeros: {zeros}/1000")
