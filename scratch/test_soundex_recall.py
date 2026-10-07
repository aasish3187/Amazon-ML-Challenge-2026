import sys
sys.path.insert(0, '.')
import time
import re
import numpy as np
from collections import defaultdict
from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
from src.normalize import normalize_record

s1_list = []
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 3 and p[3] == 'India':
            s1_list.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3]))
            if len(s1_list) >= 500:
                break

gt_map = {}
s1_ids = {r['entity_id'] for r in s1_list}
needed_matches = set()
with open(TRAIN_GT, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] in s1_ids:
            m = set(p[1].split(',')) if len(p)>1 and p[1] else set()
            gt_map[p[0]] = m
            needed_matches.update(m)

s2_pool = []
s3_pool = []
import random
random.seed(42)
with open(TRAIN_S2, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 3 and p[3] == 'India':
            if p[0] in needed_matches or (len(s2_pool) < 40000 and random.random() < 0.05):
                s2_pool.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3]))

with open(TRAIN_S3, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 3 and p[3] == 'India':
            if p[0] in needed_matches or (len(s3_pool) < 40000 and random.random() < 0.05):
                s3_pool.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3]))

def soundex(name):
    if not name or not name.isalpha(): return ''
    name = name.upper()
    codes = {'B':'1','F':'1','P':'1','V':'1',
             'C':'2','G':'2','J':'2','K':'2','Q':'2','S':'2','X':'2','Z':'2',
             'D':'3','T':'3','L':'4','M':'5','N':'5','R':'6'}
    first = name[0]
    tail = [codes.get(c, '') for c in name[1:]]
    res = [first]
    prev = codes.get(first, '')
    for c in tail:
        if c != prev and c != '':
            res.append(c)
        prev = c
    return (''.join(res) + '0000')[:4]

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}
COMMON_ADDR_STOP = {
    'the', 'and', 'road', 'rd', 'street', 'st', 'lane', 'ln', 'floor', 'fl',
    'building', 'bldg', 'near', 'opp', 'opposite', 'behind', 'beside', 'plot',
    'no', 'hno', 'dno', 'door', 'shop', 'flat', 'apartment', 'nagar', 'colony',
    'india', 'delhi', 'mumbai', 'bengaluru', 'hyderabad', 'chennai', 'kolkata',
    'state', 'city', 'west', 'east', 'north', 'south', 'cross', 'main', 'sector'
}

for r in s1_list + s2_pool + s3_pool:
    r['soundex_tokens'] = [soundex(tok) for tok in r['name_tokens'] if len(tok) >= 3 and tok not in STOPWORDS]

def index_source(records, max_len=600):
    t_idx = defaultdict(list)
    p_idx = defaultdict(list)
    num_idx = defaultdict(list)
    addr_tok_idx = defaultdict(list)
    sx_idx = defaultdict(list)
    for idx, r in enumerate(records):
        for tok in r['name_tokens']:
            if len(tok) >= 3 and tok not in STOPWORDS and len(t_idx[tok]) < max_len:
                t_idx[tok].append(idx)
        if len(r['norm_name']) >= 4 and len(p_idx[r['norm_name'][:4]]) < max_len:
            p_idx[r['norm_name'][:4]].append(idx)
        for num in r['addr_numbers']:
            if len(num) >= 2 and len(num_idx[num]) < max_len:
                num_idx[num].append(idx)
        for tok in r['norm_addr'].split():
            if len(tok) >= 5 and tok not in COMMON_ADDR_STOP and len(addr_tok_idx[tok]) < max_len:
                addr_tok_idx[tok].append(idx)
        for sx in r['soundex_tokens']:
            if sx and len(sx_idx[sx]) < max_len:
                sx_idx[sx].append(idx)
    return (t_idx, p_idx, num_idx, addr_tok_idx, sx_idx)

s2_idxs = index_source(s2_pool)
s3_idxs = index_source(s3_pool)

def query_candidates(s1_rec, pool_recs, idxs, top_k=25):
    t_idx, p_idx, num_idx, addr_tok_idx, sx_idx = idxs
    sc = defaultdict(int)
    # Name tokens
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok not in STOPWORDS:
            w = 8 if len(tok) >= 5 else 5
            for pi in t_idx.get(tok, ()):
                sc[pi] += w
    # Soundex phonetic tokens (catches transliterations!)
    for sx in s1_rec['soundex_tokens']:
        if sx:
            for pi in sx_idx.get(sx, ()):
                sc[pi] += 4
    # Name prefix
    if len(s1_rec['norm_name']) >= 4:
        for pi in p_idx.get(s1_rec['norm_name'][:4], ()):
            sc[pi] += 4
    # Address numbers
    for num in s1_rec['addr_numbers']:
        if len(num) >= 2:
            w = 12 if len(num) == 6 else 4
            for pi in num_idx.get(num, ()):
                sc[pi] += w
    # Address tokens
    for tok in s1_rec['norm_addr'].split():
        if len(tok) >= 5 and tok not in COMMON_ADDR_STOP:
            for pi in addr_tok_idx.get(tok, ()):
                sc[pi] += 5

    if not sc: return []
    return [pi for pi, _ in sorted(sc.items(), key=lambda x: -x[1])[:top_k]]

total_true_s2 = sum(len([x for x in v if x.startswith('S2-')]) for v in gt_map.values())
total_true_s3 = sum(len([x for x in v if x.startswith('S3-')]) for v in gt_map.values())
found_s2 = 0
found_s3 = 0

for s1_rec in s1_list:
    s1_id = s1_rec['entity_id']
    gt_s = gt_map.get(s1_id, set())
    c2 = query_candidates(s1_rec, s2_pool, s2_idxs, top_k=25)
    c3 = query_candidates(s1_rec, s3_pool, s3_idxs, top_k=25)
    c2_ids = {s2_pool[pi]['entity_id'] for pi in c2}
    c3_ids = {s3_pool[pi]['entity_id'] for pi in c3}
    for t in gt_s:
        if t in c2_ids: found_s2 += 1
        if t in c3_ids: found_s3 += 1

print(f"RESULTS WITH SOUNDEX PHONETIC MATCHING:")
print(f"  S2 Recall: {found_s2}/{total_true_s2} = {found_s2/total_true_s2*100:.2f}%")
print(f"  S3 Recall: {found_s3}/{total_true_s3} = {found_s3/total_true_s3*100:.2f}%")
