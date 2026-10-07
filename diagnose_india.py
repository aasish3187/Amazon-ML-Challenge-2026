import os
import sys
from collections import defaultdict
from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
from src.normalize import normalize_record
from src.blocking import run_full_blocking

sys.stdout.reconfigure(encoding='utf-8')

# Read 300 India S1 records
s1_recs = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 3 and parts[3] == 'India':
            rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3])
            rec['raw_name'] = parts[1]
            rec['raw_addr'] = parts[2] if len(parts) > 2 else ''
            s1_recs.append(rec)
            if len(s1_recs) >= 300:
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
                rec = normalize_record(parts[0], parts[1], parts[2] if len(parts) > 2 else '', parts[3] if len(parts) > 3 else '')
                rec['raw_name'] = parts[1]
                rec['raw_addr'] = parts[2] if len(parts) > 2 else ''
                pool_recs.append(rec)

cands = run_full_blocking(s1_recs, pool_recs, top_k=50, max_candidates=100)

print('\n=== MISSED MATCHES DIAGNOSIS (INDIA) ===')
missed_count = 0
for s1_idx, s1_rec in enumerate(s1_recs):
    s1_id = s1_rec['entity_id']
    true_m = target_gt.get(s1_id, set())
    cand_indices = cands.get(s1_idx, set())
    cand_ids = {pool_recs[ci]['entity_id'] for ci in cand_indices}
    for tm in true_m:
        if tm not in cand_ids:
            tm_rec = next((r for r in pool_recs if r['entity_id'] == tm), None)
            if tm_rec and missed_count < 15:
                missed_count += 1
                print(f"\n--- Missed #{missed_count} ---")
                print(f"S1 ID={s1_id}")
                print(f"  Raw Name : {s1_rec['raw_name']}")
                print(f"  Norm Name: {s1_rec['norm_name']}")
                print(f"  Raw Addr : {s1_rec['raw_addr']}")
                print(f"  Norm Addr: {s1_rec['norm_addr']}")
                print(f"Match ID={tm}")
                print(f"  Raw Name : {tm_rec['raw_name']}")
                print(f"  Norm Name: {tm_rec['norm_name']}")
                print(f"  Raw Addr : {tm_rec['raw_addr']}")
                print(f"  Norm Addr: {tm_rec['norm_addr']}")
