import sys, os, time, re, random
from collections import defaultdict, Counter

sys.path.insert(0, '.')
from src.normalize import normalize_record

# Load pool from partitions/test/India/pool.tsv or train
print("Testing index and scoring on India training sample...")
# We will load 500 S1 records and 500,000 pool records from train_source2 + train_source3
gt = {}
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 1 and p[1].strip():
            gt[p[0]] = set(p[1].split(','))

s1_list = []
needed_m = set()
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 3 and p[3] == 'India':
            m = gt.get(p[0], set())
            if m:
                s1_list.append((normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3]), m))
                needed_m.update(m)
                if len(s1_list) >= 200:
                    break

print(f"Loaded {len(s1_list)} S1 entities with {len(needed_m)} true matches.")

# Load pool: all needed matches + 300,000 random pool records to simulate a real pool
pool_records = []
pool_id_to_idx = {}
loaded_needed = 0

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}
COMMON_ADDR = {'road', 'street', 'lane', 'floor', 'building', 'near', 'opposite', 'behind', 'plot', 'shop', 'flat', 'apartment', 'nagar', 'colony', 'india', 'state', 'city', 'cross', 'main', 'sector'}

for path in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            p = line.strip().split('\t')
            if len(p) > 3 and p[3] == 'India':
                if p[0] in needed_m or len(pool_records) < 300000:
                    idx = len(pool_records)
                    rec = normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3])
                    pool_records.append(rec)
                    pool_id_to_idx[p[0]] = idx
                    if p[0] in needed_m:
                        loaded_needed += 1
            if len(pool_records) >= 300000 and loaded_needed >= len(needed_m):
                break

print(f"Pool constructed: {len(pool_records):,} records ({loaded_needed}/{len(needed_m)} needed matches included).")

# Now let's test different indexing & retrieval strategies
# Strategy A: Standard TF-IDF / IDF-weighted inverted index
# Compute document frequency (DF) of tokens across pool
name_df = Counter()
addr_df = Counter()
num_df = Counter()

for rec in pool_records:
    for t in rec['name_tokens']:
        if len(t) >= 3 and t not in STOPWORDS:
            name_df[t] += 1
    for num in rec['addr_numbers']:
        if len(num) >= 2:
            num_df[num] += 1
    for t in rec['norm_addr'].split():
        if len(t) >= 4 and t not in COMMON_ADDR:
            addr_df[t] += 1

N = len(pool_records)
import math

def idf(df):
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

# Build inverted indexes (capped vs uncapped)
name_idx = defaultdict(list)
num_idx = defaultdict(list)
addr_idx = defaultdict(list)

for idx, rec in enumerate(pool_records):
    # Only index tokens that aren't ultra-common (DF < 5% of pool)
    for t in rec['name_tokens']:
        if len(t) >= 3 and t not in STOPWORDS and name_df[t] < 0.05 * N:
            name_idx[t].append(idx)
    for num in rec['addr_numbers']:
        if len(num) >= 2 and num_df[num] < 0.05 * N:
            num_idx[num].append(idx)
    for t in rec['norm_addr'].split():
        if len(t) >= 4 and t not in COMMON_ADDR and addr_df[t] < 0.05 * N:
            addr_idx[t].append(idx)

print(f"Index built: names={len(name_idx)}, nums={len(num_idx)}, addrs={len(addr_idx)}")

# Evaluate Recall @ K for different K
for K in [15, 25, 50, 100]:
    hits = 0
    total = 0
    for s1_rec, m_set in s1_list:
        valid_m = {m for m in m_set if m in pool_id_to_idx}
        if not valid_m:
            continue
        total += len(valid_m)
        
        # Query
        scores = defaultdict(float)
        # Score by IDF: rarer tokens give huge score!
        for t in s1_rec['name_tokens']:
            if t in name_idx:
                score = idf(name_df[t]) * 2.0
                postings = name_idx[t]
                if len(postings) > 2000:
                    postings = postings[:2000]
                for pi in postings:
                    scores[pi] += score
                    
        for num in s1_rec['addr_numbers']:
            if num in num_idx:
                # 6 digit pincode is golden
                weight = 3.0 if len(num) == 6 else 1.5
                score = idf(num_df[num]) * weight
                postings = num_idx[num]
                if len(postings) > 2000:
                    postings = postings[:2000]
                for pi in postings:
                    scores[pi] += score
                    
        for t in s1_rec['norm_addr'].split():
            if t in addr_idx:
                score = idf(addr_df[t]) * 1.5
                postings = addr_idx[t]
                if len(postings) > 2000:
                    postings = postings[:2000]
                for pi in postings:
                    scores[pi] += score
                    
        top_k = sorted(scores.items(), key=lambda x: -x[1])[:K]
        cand_ids = {pool_records[pi]['entity_id'] for pi, _ in top_k}
        
        hits += len(cand_ids & valid_m)
        
    print(f"Recall @ {K:3d}: {hits:4d} / {total:4d} ({hits/total*100:.2f}%)")
