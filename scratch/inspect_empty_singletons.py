import sys, os
sys.path.insert(0, '.')
from src.normalize import normalize_record

# Find 10 S1 entities that have empty matches in matching_India.tsv
empty_eids = []
with open('output/matching_India.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) <= 1 or not p[1].strip():
            empty_eids.append(p[0])
            if len(empty_eids) >= 10:
                break

print(f"Sample 10 empty S1 IDs from India: {empty_eids}")

# Get their details from partitions/test/India/s1.tsv
s1_details = {}
with open('partitions/test/India/s1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] in empty_eids:
            s1_details[p[0]] = (p[1], p[2] if len(p)>2 else '')

# Get their candidate lists from output/candidate_India.tsv
cand_map = {}
with open('output/candidate_India.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] in empty_eids:
            cand_map[p[0]] = p[1].split(',') if len(p)>1 and p[1] else []

# Inspect candidates in pool
needed_cids = set()
for c_list in cand_map.values():
    needed_cids.update(c_list)

pool_details = {}
with open('partitions/test/India/pool.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] in needed_cids:
            pool_details[p[0]] = (p[1], p[2] if len(p)>2 else '')
            if len(pool_details) == len(needed_cids):
                break

for eid in empty_eids:
    s1_name, s1_addr = s1_details.get(eid, ('', ''))
    cands = cand_map.get(eid, [])
    print("=" * 70)
    print(f"S1: {eid} | '{s1_name}' | '{s1_addr}'")
    print(f"  Candidates generated ({len(cands)} total):")
    for cid in cands[:6]:
        cn, ca = pool_details.get(cid, ('', ''))
        print(f"    - {cid} | '{cn}' | '{ca}'")
