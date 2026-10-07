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
            if len(s1_list) >= 200:
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
with open(TRAIN_S2, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] in needed_matches:
            s2_pool.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3] if len(p)>3 else ''))

with open(TRAIN_S3, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] in needed_matches:
            s3_pool.append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3] if len(p)>3 else ''))

print(f"Loaded {len(s1_list)} S1, {len(s2_pool)} S2 matches, {len(s3_pool)} S3 matches.")

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}

def extract_pin(addr):
    m = re.findall(r'\b[1-9][0-9]{5}\b', addr)
    return m[0] if m else ''

for r in s1_list + s2_pool + s3_pool:
    r['pincode'] = extract_pin(r['norm_addr'])

# Check what tokens/features match
s2_map = {r['entity_id']: r for r in s2_pool}
s3_map = {r['entity_id']: r for r in s3_pool}
pool_all = {**s2_map, **s3_map}

missed = []
for s1 in s1_list:
    for mid in gt_map.get(s1['entity_id'], []):
        mrec = pool_all.get(mid)
        if not mrec: continue
        # Check token matches
        t1 = set(s1['name_tokens']) - STOPWORDS
        t2 = set(mrec['name_tokens']) - STOPWORDS
        overlap = t1 & t2
        pin_match = (s1['pincode'] == mrec['pincode']) if (s1['pincode'] and mrec['pincode']) else False
        p1 = s1['norm_name'][:4]
        p2 = mrec['norm_name'][:4]
        pref_match = (p1 == p2) if (len(p1)==4 and len(p2)==4) else False
        
        if not overlap and not pref_match and not pin_match:
            missed.append((s1, mrec))

print(f"Pairs with NO name overlap, NO prefix, NO pin match: {len(missed)}")
for s1, mrec in missed[:10]:
    print(f"---")
    print(f"S1:   {s1['entity_id']} | '{s1['norm_name']}' | '{s1['norm_addr']}'")
    print(f"Match:{mrec['entity_id']} | '{mrec['norm_name']}' | '{mrec['norm_addr']}'")
