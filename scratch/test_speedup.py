import sys, os, time, math
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, '.')
from src.normalize import normalize_record
from src.fast_predict import soundex

# Load 1,000 S1 and 100k S2 pool
s1_recs = []
with open('partitions/test/France/s1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        s1_recs.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', 'France'))
        if len(s1_recs) >= 1000:
            break

pool_records = []
with open('partitions/test/France/pool.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 3 and p[0].startswith('S2-'):
            pool_records.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', 'France'))
            if len(pool_records) >= 300000:
                break

N = len(pool_records)
name_df = Counter()
addr_df = Counter()
num_df = Counter()

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}
COMMON_ADDR = {'road', 'street', 'lane', 'floor', 'building', 'near', 'opposite', 'behind', 'france', 'rue', 'avenue', 'boulevard', 'allee', 'impasse', 'chemin'}

for rec in pool_records:
    for t in rec['name_tokens']:
        if len(t) >= 3 and t not in STOPWORDS: name_df[t] += 1
    for num in rec['addr_numbers']: num_df[num] += 1
    for t in rec['norm_addr'].split():
        if len(t) >= 3 and t not in COMMON_ADDR: addr_df[t] += 1

# Filter DF < 0.01 * N (1% threshold instead of 5%)
name_idx = defaultdict(list)
num_idx = defaultdict(list)
addr_idx = defaultdict(list)

MAX_POSTINGS = 300 # Cap at 300 instead of 1500

for idx, rec in enumerate(pool_records):
    for t in rec['name_tokens']:
        if len(t) >= 3 and t not in STOPWORDS and name_df[t] < 0.01 * N:
            if len(name_idx[t]) < MAX_POSTINGS:
                name_idx[t].append(idx)
    for num in rec['addr_numbers']:
        if num_df[num] < 0.01 * N:
            if len(num_idx[num]) < MAX_POSTINGS:
                num_idx[num].append(idx)
    for t in rec['norm_addr'].split():
        if len(t) >= 3 and t not in COMMON_ADDR and addr_df[t] < 0.01 * N:
            if len(addr_idx[t]) < MAX_POSTINGS:
                addr_idx[t].append(idx)

def idf(df):
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

t0 = time.time()
for s1_rec in s1_recs:
    scores = defaultdict(float)
    for t in s1_rec['name_tokens']:
        if t in name_idx:
            sc = idf(name_df[t]) * 3.0
            for pi in name_idx[t]:
                scores[pi] += sc
    for num in s1_rec['addr_numbers']:
        if num in num_idx:
            sc = idf(num_df[num]) * (3.5 if len(num)==5 else 1.5)
            for pi in num_idx[num]:
                scores[pi] += sc
    for t in s1_rec['norm_addr'].split():
        if t in addr_idx:
            sc = idf(addr_df[t]) * 1.5
            for pi in addr_idx[t]:
                scores[pi] += sc
    top_15 = sorted(scores.items(), key=lambda x: -x[1])[:15]

elapsed = time.time() - t0
print(f"Queried 1,000 entities in {elapsed:.3f}s ({1000/elapsed:.0f} S1/sec)")
