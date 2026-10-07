import os
import sys
import time
from collections import defaultdict
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.config import *
from src.normalize import normalize_record
from src.evaluate import parse_ground_truth

print("Testing blocking recall on India training data...")

gt = parse_ground_truth(TRAIN_GT)

# Load 1,000 India S1 entities that have matches
s1_recs = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 3 and parts[3].strip() == 'India':
            eid = parts[0]
            if len(gt.get(eid, set())) > 0:  # has true matches
                s1_recs.append(normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3]))
                if len(s1_recs) >= 1000:
                    break

print(f"Loaded {len(s1_recs)} S1 records with true matches.")

# Now let's see how many of their true matches are in S2 vs S3
s1_ids = {r['entity_id'] for r in s1_recs}
needed_true = set()
for r in s1_recs:
    needed_true.update(gt[r['entity_id']])

print(f"Total true matches needed: {len(needed_true)}")

# Let's test the current fast_predict.py indexing logic with MAX_INDEX_LEN=600 vs without cap vs with TF-IDF
# First load 100,000 pool records (including needed_true) from India train S2+S3
pool_records = []
import random
random.seed(42)

for p in [TRAIN_S2, TRAIN_S3]:
    with open(p, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 3 and parts[3].strip() == 'India':
                eid = parts[0]
                if eid in needed_true:
                    pool_records.append(normalize_record(eid, parts[1], parts[2] if len(parts) > 2 else '', 'India'))
                elif len(pool_records) < 150000 and random.random() < 0.1:
                    pool_records.append(normalize_record(eid, parts[1], parts[2] if len(parts) > 2 else '', 'India'))

print(f"Constructed pool with {len(pool_records)} records ({len(needed_true)} true matches included).")

# Test 1: Current fast_predict index (with MAX_INDEX_LEN=600)
from src.fast_predict import soundex, STOPWORDS, COMMON_ADDR_STOP

def test_indexer(max_len=600):
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

    for pool_idx, rec in enumerate(pool_records):
        is_s2 = rec['entity_id'].startswith('S2-')
        name_idx = s2_name_idx if is_s2 else s3_name_idx
        pref_idx = s2_pref_idx if is_s2 else s3_pref_idx
        num_idx = s2_num_idx if is_s2 else s3_num_idx
        addr_idx = s2_addr_idx if is_s2 else s3_addr_idx
        sx_idx = s2_sx_idx if is_s2 else s3_sx_idx

        for tok in rec['name_tokens']:
            if len(tok) >= 3 and tok not in STOPWORDS:
                if max_len is None or len(name_idx[tok]) < max_len:
                    name_idx[tok].append(pool_idx)

        if len(rec['norm_name']) >= 4:
            pref = rec['norm_name'][:4]
            if max_len is None or len(pref_idx[pref]) < max_len:
                pref_idx[pref].append(pool_idx)

        for num in rec['addr_numbers']:
            if len(num) >= 2:
                if max_len is None or len(num_idx[num]) < max_len:
                    num_idx[num].append(pool_idx)

        for tok in rec['norm_addr'].split():
            if len(tok) >= 5 and tok not in COMMON_ADDR_STOP:
                if max_len is None or len(addr_idx[tok]) < max_len:
                    addr_idx[tok].append(pool_idx)

        for tok in rec['name_tokens']:
            if len(tok) >= 3 and tok not in STOPWORDS:
                sx = soundex(tok)
                if sx:
                    if max_len is None or len(sx_idx[sx]) < max_len:
                        sx_idx[sx].append(pool_idx)

    # Now evaluate candidate recall
    from src.fast_predict import query_source_candidates
    s2_idxs = (s2_name_idx, s2_pref_idx, s2_num_idx, s2_addr_idx, s2_sx_idx)
    s3_idxs = (s3_name_idx, s3_pref_idx, s3_num_idx, s3_addr_idx, s3_sx_idx)

    found_matches = 0
    total_matches = 0
    zero_cands = 0

    for s1_rec in s1_recs:
        true_m = gt[s1_rec['entity_id']]
        c2 = query_source_candidates(s1_rec, s2_idxs, True, top_k=25)
        c3 = query_source_candidates(s1_rec, s3_idxs, True, top_k=25)
        cand_eids = {pool_records[pi]['entity_id'] for pi in c2 + c3}

        if len(cand_eids) == 0:
            zero_cands += 1

        for tm in true_m:
            if tm in cand_eids:
                found_matches += 1
            total_matches += 1

    print(f"Max Len={max_len}: Found {found_matches}/{total_matches} ({found_matches/total_matches*100:.2f}% recall). Zero cand entities: {zero_cands}/{len(s1_recs)} ({zero_cands/len(s1_recs)*100:.1f}%)")

print("\n--- Running with MAX_INDEX_LEN = 600 ---")
test_indexer(600)

print("\n--- Running with MAX_INDEX_LEN = 5000 ---")
test_indexer(5000)

print("\n--- Running with NO CAP (max_len = None) ---")
test_indexer(None)
