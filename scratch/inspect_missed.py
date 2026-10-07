import re
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
needed = set()
with open(TRAIN_GT, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[0] in s1_ids:
            m = set(p[1].split(',')) if len(p)>1 and p[1] else set()
            gt_map[p[0]] = m
            needed.update(m)

pool = {}
for path in [TRAIN_S2, TRAIN_S3]:
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if p[0] in needed:
                pool[p[0]] = normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}

count = 0
for s1 in s1_list:
    s1_id = s1['entity_id']
    for mid in gt_map.get(s1_id, []):
        mrec = pool.get(mid)
        if not mrec: continue
        shared_name_tokens = (set(s1['name_tokens']) & set(mrec['name_tokens'])) - STOPWORDS
        shared_prefix = s1['norm_name'][:3] == mrec['norm_name'][:3] if len(s1['norm_name'])>=3 and len(mrec['norm_name'])>=3 else False
        shared_nums = set(s1['addr_numbers']) & set(mrec['addr_numbers'])
        shared_addr_tokens = (set(s1['norm_addr'].split()) & set(mrec['norm_addr'].split())) - STOPWORDS
        
        if not shared_name_tokens and not shared_prefix and not shared_nums and not shared_addr_tokens:
            print(f"MISSED MATCH:")
            print(f"  S1:   {s1['entity_id']} | '{s1['business_name']}' | '{s1['business_address']}'")
            print(f"  Match:{mrec['entity_id']} | '{mrec['business_name']}' | '{mrec['business_address']}'")
            count += 1
            if count >= 5: break
    if count >= 5: break

print(f"Total pure misses out of evaluated: {count}")
