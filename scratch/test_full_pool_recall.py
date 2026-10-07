import os
import sys
import time
import math
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.config import *
from src.normalize import normalize_record
from src.evaluate import parse_ground_truth
from src.fast_predict import soundex, STOPWORDS, COMMON_ADDR_STOP

print("=" * 70)
print("TESTING FULL-POOL CANDIDATE RECALL ON INDIA")
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

print(f"Loaded {len(s1_recs):,} S1 evaluation records.")

# Index ALL of Train S2 and S3 for India
t0 = time.time()
pool_records = []
s2_name_idx = defaultdict(list)
s2_num_idx = defaultdict(list)
s2_pref_idx = defaultdict(list)
s2_sx_idx = defaultdict(list)
s2_ngram_idx = defaultdict(list)

s3_name_idx = defaultdict(list)
s3_num_idx = defaultdict(list)
s3_pref_idx = defaultdict(list)
s3_sx_idx = defaultdict(list)
s3_ngram_idx = defaultdict(list)

n = 0
for path in [TRAIN_S2, TRAIN_S3]:
    is_s2 = (path == TRAIN_S2)
    name_idx = s2_name_idx if is_s2 else s3_name_idx
    num_idx = s2_num_idx if is_s2 else s3_num_idx
    pref_idx = s2_pref_idx if is_s2 else s3_pref_idx
    sx_idx = s2_sx_idx if is_s2 else s3_sx_idx
    ngram_idx = s2_ngram_idx if is_s2 else s3_ngram_idx

    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 3 and parts[3].strip() == 'India':
                rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', 'India')
                pi = len(pool_records)
                pool_records.append(rec)

                # 1. Name tokens
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

                # 4. Character 3-grams of name (e.g. for typo tolerance)
                # Only keep distinct 3-grams
                name_clean = rec['norm_name'].replace(' ', '')
                if len(name_clean) >= 3:
                    ngs = {name_clean[i:i+3] for i in range(len(name_clean)-2)}
                    for ng in ngs:
                        ngram_idx[ng].append(pi)

                n += 1
                if n % 1000000 == 0:
                    print(f"  Indexed {n:,} pool records in {time.time()-t0:.1f}s...")

print(f"Full pool indexing complete: {len(pool_records):,} records in {time.time()-t0:.1f}s.")

# Compute IDF for tokens and numbers to weight matches properly
N_pool = len(pool_records)
def compute_idfs(idx):
    idfs = {}
    for k, postings in idx.items():
        # Standard BM25-style IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df = len(postings)
        idfs[k] = math.log(N_pool / (df + 1.0))
    return idfs

print("Computing IDFs...")
s2_name_idf = compute_idfs(s2_name_idx)
s3_name_idf = compute_idfs(s3_name_idx)
s2_num_idf = compute_idfs(s2_num_idx)
s3_num_idf = compute_idfs(s3_num_idx)

# Test candidate generation with IDF scoring
def query_with_idf(s1_rec, indexes, idf_maps, top_k=25, max_postings=2000):
    name_idx, pref_idx, num_idx, sx_idx, ngram_idx = indexes
    name_idf, num_idf = idf_maps
    sc = defaultdict(float)

    # 1. Name tokens with IDF weighting
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok in name_idx:
            postings = name_idx[tok]
            if len(postings) <= max_postings:
                idf = name_idf.get(tok, 1.0)
                # Rare words give huge boost, common words give smaller boost
                w = idf * 2.0
                for pi in postings:
                    sc[pi] += w

    # 2. 4-char prefix
    if len(s1_rec['norm_name']) >= 4:
        pref = s1_rec['norm_name'][:4]
        if pref in pref_idx:
            postings = pref_idx[pref]
            if len(postings) <= max_postings:
                for pi in postings:
                    sc[pi] += 4.0

    # 3. Address numbers (PIN codes, house numbers) with IDF
    for num in s1_rec['addr_numbers']:
        if len(num) >= 2 and num in num_idx:
            postings = num_idx[num]
            if len(postings) <= max_postings:
                idf = num_idf.get(num, 1.0)
                w = idf * 2.5 if len(num) == 6 else idf * 1.0
                for pi in postings:
                    sc[pi] += w

    # 4. Soundex
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok not in STOPWORDS:
            sx = soundex(tok)
            if sx and sx in sx_idx:
                postings = sx_idx[sx]
                if len(postings) <= 1000:
                    for pi in postings:
                        sc[pi] += 2.0

    if not sc:
        return []
    # Return top_k highest scoring candidates
    if len(sc) <= top_k:
        return list(sc.keys())
    # Partial sort for speed
    items = list(sc.items())
    top_items = sorted(items, key=lambda x: -x[1])[:top_k]
    return [pi for pi, _ in top_items]

s2_idxs = (s2_name_idx, s2_pref_idx, s2_num_idx, s2_sx_idx, s2_ngram_idx)
s3_idxs = (s3_name_idx, s3_pref_idx, s3_num_idx, s3_sx_idx, s3_ngram_idx)
s2_idfs = (s2_name_idf, s2_num_idf)
s3_idfs = (s3_name_idf, s3_num_idf)

print("\nEvaluating candidate recall on 2,000 S1 entities across full pool...")
t_eval = time.time()
found_matches = 0
total_matches = 0
zero_cands = 0
total_cands = 0

for s1_rec in s1_recs:
    true_m = gt[s1_rec['entity_id']]
    c2 = query_with_idf(s1_rec, s2_idxs, s2_idfs, top_k=25, max_postings=3000)
    c3 = query_with_idf(s1_rec, s3_idxs, s3_idfs, top_k=25, max_postings=3000)
    cand_eids = {pool_records[pi]['entity_id'] for pi in c2 + c3}

    total_cands += len(cand_eids)
    if len(cand_eids) == 0:
        zero_cands += 1

    for tm in true_m:
        if tm in cand_eids:
            found_matches += 1
        total_matches += 1

elapsed = time.time() - t_eval
print(f"Results across {len(s1_recs):,} entities in {elapsed:.1f}s ({len(s1_recs)/elapsed:.0f} S1/sec):")
print(f"  Candidate recall: {found_matches}/{total_matches} ({found_matches/total_matches*100:.2f}%)")
print(f"  Zero candidate entities: {zero_cands}/{len(s1_recs)} ({zero_cands/len(s1_recs)*100:.2f}%)")
print(f"  Avg candidates per entity: {total_cands/len(s1_recs):.1f}")
