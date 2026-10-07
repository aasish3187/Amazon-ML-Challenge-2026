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
from src.evaluate import compute_f05_per_entity, parse_ground_truth
from src.train_ensemble import EnsemblePredictor
from src.fast_predict import soundex, STOPWORDS, COMMON_ADDR_STOP
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

print("=" * 70)
print("TESTING INDIA CANDIDATE BOOST: CHAR N-GRAMS + TRANSITIVE GRAPH")
print("=" * 70)

gt = parse_ground_truth(TRAIN_GT)

# Load 2,000 India S1 entities with true matches
s1_recs = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 3 and parts[3].strip() == 'India':
            eid = parts[0]
            if len(gt.get(eid, set())) > 0:
                s1_recs.append(normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', 'India'))
                if len(s1_recs) >= 2000:
                    break

s1_ids = {r['entity_id'] for r in s1_recs}
needed_matches = set()
for r in s1_recs:
    needed_matches.update(gt.get(r['entity_id'], set()))

print(f"Loaded {len(s1_recs):,} S1 entities. True matches needed: {len(needed_matches):,}")

# Load pool: true matches + 80,000 distractors
import random
random.seed(42)
pool_records = []
for path in [TRAIN_S2, TRAIN_S3]:
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 3 and parts[3].strip() == 'India':
                eid = parts[0]
                if eid in needed_matches:
                    pool_records.append(normalize_record(eid, parts[1], parts[2] if len(parts) > 2 else '', 'India'))
                elif len(pool_records) < 100000 and random.random() < 0.04:
                    pool_records.append(normalize_record(eid, parts[1], parts[2] if len(parts) > 2 else '', 'India'))

print(f"Pool size: {len(pool_records):,} records")

# Build indexes with char 3-grams
s2_name_idx = defaultdict(list)
s2_pref_idx = defaultdict(list)
s2_num_idx = defaultdict(list)
s2_sx_idx = defaultdict(list)
s2_c3_idx = defaultdict(list)

s3_name_idx = defaultdict(list)
s3_pref_idx = defaultdict(list)
s3_num_idx = defaultdict(list)
s3_sx_idx = defaultdict(list)
s3_c3_idx = defaultdict(list)

MAX_LEN = 3000

for pi, rec in enumerate(pool_records):
    is_s2 = rec['entity_id'].startswith('S2-')
    name_idx = s2_name_idx if is_s2 else s3_name_idx
    pref_idx = s2_pref_idx if is_s2 else s3_pref_idx
    num_idx = s2_num_idx if is_s2 else s3_num_idx
    sx_idx = s2_sx_idx if is_s2 else s3_sx_idx
    c3_idx = s2_c3_idx if is_s2 else s3_c3_idx

    for tok in rec['name_tokens']:
        if len(tok) >= 3 and tok not in STOPWORDS and len(name_idx[tok]) < MAX_LEN:
            name_idx[tok].append(pi)
        sx = soundex(tok)
        if sx and len(sx_idx[sx]) < MAX_LEN:
            sx_idx[sx].append(pi)

    if len(rec['norm_name']) >= 3 and len(pref_idx[rec['norm_name'][:3]]) < MAX_LEN:
        pref_idx[rec['norm_name'][:3]].append(pi)

    for num in rec['addr_numbers']:
        if len(num) >= 2 and len(num_idx[num]) < MAX_LEN:
            num_idx[num].append(pi)

    # Distinct char 3-grams of name
    clean_name = rec['norm_name'].replace(' ', '')
    if len(clean_name) >= 3:
        for i in range(len(clean_name)-2):
            g3 = clean_name[i:i+3]
            if len(c3_idx[g3]) < 1000:
                c3_idx[g3].append(pi)

def query_enhanced(s1_rec, indexes, top_k=30):
    name_idx, pref_idx, num_idx, sx_idx, c3_idx = indexes
    sc = defaultdict(float)

    # 1. Name tokens
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok in name_idx:
            w = 8.0 if len(tok) >= 5 else 5.0
            for pi in name_idx[tok]:
                sc[pi] += w

    # 2. Soundex
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok not in STOPWORDS:
            sx = soundex(tok)
            if sx in sx_idx:
                for pi in sx_idx[sx]:
                    sc[pi] += 4.0

    # 3. 3-char prefix
    if len(s1_rec['norm_name']) >= 3:
        pref = s1_rec['norm_name'][:3]
        if pref in pref_idx:
            for pi in pref_idx[pref]:
                sc[pi] += 4.0

    # 4. Address numbers (PIN codes are huge!)
    for num in s1_rec['addr_numbers']:
        if len(num) >= 2 and num in num_idx:
            w = 12.0 if len(num) == 6 else 4.0
            for pi in num_idx[num]:
                sc[pi] += w

    # 5. Char 3-grams (typo / transliteration tolerance)
    clean_name = s1_rec['norm_name'].replace(' ', '')
    if len(clean_name) >= 3:
        for i in range(len(clean_name)-2):
            g3 = clean_name[i:i+3]
            if g3 in c3_idx:
                for pi in c3_idx[g3]:
                    sc[pi] += 0.8

    if not sc:
        return []
    return [pi for pi, _ in sorted(sc.items(), key=lambda x: -x[1])[:top_k]]

s2_idxs = (s2_name_idx, s2_pref_idx, s2_num_idx, s2_sx_idx, s2_c3_idx)
s3_idxs = (s3_name_idx, s3_pref_idx, s3_num_idx, s3_sx_idx, s3_c3_idx)

hits = 0
total = 0
for s1_rec in s1_recs:
    true_set = gt.get(s1_rec['entity_id'], set())
    c2 = query_enhanced(s1_rec, s2_idxs, top_k=30)
    c3 = query_enhanced(s1_rec, s3_idxs, top_k=30)
    cand_eids = {pool_records[pi]['entity_id'] for pi in c2 + c3}

    for tm in true_set:
        if tm in cand_eids:
            hits += 1
        total += 1

print(f"\nEnhanced Candidate Recall on India: {hits}/{total} ({hits/total*100:.2f}%)")
print(f"Compare: previous India recall was 91.58%. Boost: +{hits/total*100 - 91.58:.2f}%!")
