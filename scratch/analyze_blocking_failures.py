import sys, os, time, re, random
from collections import defaultdict, Counter

sys.path.insert(0, '.')
from src.normalize import normalize_record, normalize_text

print("Loading 1,000 India S1 entities and ground truth...")
gt = {}
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 1 and p[1].strip():
            gt[p[0]] = set(p[1].split(','))
        else:
            gt[p[0]] = set()

s1_sample = []
s1_needed_matches = set()
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(p) > 3 and p[3] == 'India':
            m = gt.get(p[0], set())
            if m: # entity with true matches
                s1_rec = normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3])
                s1_sample.append((s1_rec, m))
                s1_needed_matches.update(m)
                if len(s1_sample) >= 500:
                    break

print(f"Sampled {len(s1_sample)} India S1 entities with {len(s1_needed_matches)} true matches.")

# Load matching pool records
pool = {}
for path in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if p[0] in s1_needed_matches:
                pool[p[0]] = normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')
                if len(pool) == len(s1_needed_matches):
                    break

print(f"Loaded {len(pool)} matching pool records.")

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}

# Now inspect why each pair would or would not be retrieved:
keys_matched = defaultdict(int)
total_pairs = 0
missed_pairs = []

for s1_rec, m_ids in s1_sample:
    for mid in m_ids:
        m_rec = pool.get(mid)
        if not m_rec:
            continue
        total_pairs += 1
        
        # Check individual blocking keys
        s1_name_toks = s1_rec['name_tokens'] - STOPWORDS
        m_name_toks = m_rec['name_tokens'] - STOPWORDS
        shared_name_toks = s1_name_toks & m_name_toks
        
        shared_nums = set(s1_rec['addr_numbers']) & set(m_rec['addr_numbers'])
        
        s1_addr_toks = {w for w in s1_rec['norm_addr'].split() if len(w) >= 4}
        m_addr_toks = {w for w in m_rec['norm_addr'].split() if len(w) >= 4}
        shared_addr_toks = s1_addr_toks & m_addr_toks
        
        shared_pref = (s1_rec['norm_name'][:4] == m_rec['norm_name'][:4]) if len(s1_rec['norm_name']) >= 4 and len(m_rec['norm_name']) >= 4 else False
        
        hit = False
        if shared_name_toks:
            keys_matched['name_token'] += 1
            hit = True
        if shared_nums:
            keys_matched['number'] += 1
            hit = True
        if shared_addr_toks:
            keys_matched['addr_token'] += 1
            hit = True
        if shared_pref:
            keys_matched['name_prefix_4'] += 1
            hit = True
            
        if hit:
            keys_matched['ANY_KEY'] += 1
        else:
            missed_pairs.append((s1_rec, m_rec))

print(f"\nTotal evaluated true pairs: {total_pairs}")
print("Theoretical Key Overlap (if index had NO caps):")
for k, v in sorted(keys_matched.items(), key=lambda x: -x[1]):
    print(f"  {k:15s}: {v:5d} / {total_pairs} ({v/total_pairs*100:.2f}%)")

print(f"\nPairs with ZERO shared keys: {len(missed_pairs)}")
for s1, m in missed_pairs[:5]:
    print("-" * 50)
    print(f"S1: {s1['entity_id']} | '{s1['norm_name']}' | '{s1['norm_addr']}' | nums: {s1['addr_numbers']}")
    print(f"M : {m['entity_id']} | '{m['norm_name']}' | '{m['norm_addr']}' | nums: {m['addr_numbers']}")
