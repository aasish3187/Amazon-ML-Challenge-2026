import os
import sys
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
from src.normalize import normalize_record
from src.blocking import run_full_blocking, tfidf_blocking

sys.stdout.reconfigure(encoding='utf-8')

# Read 200 India S1 records
s1_recs = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 3 and parts[3] == 'India':
            s1_recs.append(normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3]))
            if len(s1_recs) >= 200:
                break

s1_ids = {r['entity_id']: r for r in s1_recs}

target_gt = {}
needed_matches = set()
with open(TRAIN_GT, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if parts[0] in s1_ids:
            matches = parts[1].split(',') if len(parts) > 1 and parts[1] else []
            target_gt[parts[0]] = set(matches)
            for m in matches:
                needed_matches.add(m)

pool_recs = []
for p in [TRAIN_S2, TRAIN_S3]:
    with open(p, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in needed_matches:
                pool_recs.append(normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3] if len(parts) > 3 else ''))

# Current blocking
cands = run_full_blocking(s1_recs, pool_recs, top_k=40, max_candidates=150)

# Add Address TF-IDF blocking
s1_addrs = [r['norm_addr'] for r in s1_recs]
pool_addrs = [r['norm_addr'] for r in pool_recs]
addr_cands = tfidf_blocking(s1_addrs, pool_addrs, [r['entity_id'] for r in pool_recs], top_k=30, ngram_range=(3, 5))

# Add Combined (Name + Address) TF-IDF blocking
s1_combined = [f"{r['norm_name']} {r['norm_addr']}" for r in s1_recs]
pool_combined = [f"{r['norm_name']} {r['norm_addr']}" for r in pool_recs]
comb_cands = tfidf_blocking(s1_combined, pool_combined, [r['entity_id'] for r in pool_recs], top_k=30, ngram_range=(3, 5))

for i in range(len(s1_recs)):
    cands[i].update(addr_cands.get(i, set()))
    cands[i].update(comb_cands.get(i, set()))

total_true = sum(len(v) for v in target_gt.values())
found_true = 0
for s1_idx, s1_rec in enumerate(s1_recs):
    s1_id = s1_rec['entity_id']
    true_m = target_gt.get(s1_id, set())
    cand_indices = cands.get(s1_idx, set())
    cand_ids = {pool_recs[ci]['entity_id'] for ci in cand_indices}
    for tm in true_m:
        if tm in cand_ids:
            found_true += 1

print(f"\n======================================================")
print(f">>> WITH ADDRESS + COMBINED BLOCKING: {found_true}/{total_true} = {found_true/total_true:.4f} <<<")
print(f"======================================================")
