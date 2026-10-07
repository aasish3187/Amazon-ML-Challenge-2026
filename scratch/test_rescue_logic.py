import sys, os, time, re
from collections import defaultdict
from rapidfuzz import fuzz

sys.path.insert(0, '.')
from src.evaluate import parse_ground_truth

gt = parse_ground_truth('student_resource/dataset/train/train_ground_truth.tsv')

STOPWORDS = {'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions', 'enterprises', 'international', 'holdings', 'global', 'technologies', 'sarl', 'sasu', 'eurl', 'association'}

def clean_name(n):
    return set(re.findall(r'[a-z0-9]{3,}', n.lower())) - STOPWORDS

def extract_nums(a):
    return set(re.findall(r'\b\d+\b', a.lower()))

# Load 5000 S1 records from US
s1_list = []
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if p[3] == 'US':
            s1_list.append((p[0], p[1], p[2] if len(p)>2 else ''))
            if len(s1_list) >= 5000:
                break

needed_matches = set()
for sid, _, _ in s1_list:
    needed_matches.update(gt.get(sid, set()))

# Load full US pool from train
print("Loading train pool records...")
pool = []
num_to_pool = defaultdict(list)
word_to_pool = defaultdict(list)

for p in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if parts[3] == 'US':
                idx = len(pool)
                eid = parts[0]
                name = parts[1]
                addr = parts[2] if len(parts)>2 else ''
                pool.append((eid, name, addr))
                nums = extract_nums(addr)
                words = clean_name(name)
                for num in nums:
                    if len(num) >= 2:
                        num_to_pool[num].append(idx)
                for w in words:
                    word_to_pool[w].append(idx)

print(f"Loaded {len(pool):,} pool records.")

# Test retrieval on the 5000 S1
tp = 0
fp = 0
fn = 0
total_gt = 0

for sid, s1_name, s1_addr in s1_list:
    true_ids = gt.get(sid, set())
    total_gt += len(true_ids)
    s1_nums = extract_nums(s1_addr)
    s1_words = clean_name(s1_name)
    
    cand_scores = defaultdict(int)
    # 1. Number candidates (super selective)
    for num in s1_nums:
        if len(num) >= 2:
            postings = num_to_pool.get(num, [])
            if len(postings) < 2000:
                for pi in postings:
                    cand_scores[pi] += 10
                    
    # 2. Word candidates
    for w in s1_words:
        postings = word_to_pool.get(w, [])
        if len(postings) < 1500:
            for pi in postings:
                cand_scores[pi] += 5

    # Filter candidates
    pred_ids = set()
    for pi, sc in cand_scores.items():
        if sc < 10:
            continue
        cid, c_name, c_addr = pool[pi]
        c_nums = extract_nums(c_addr)
        # Rule 1: No conflicting numbers
        if s1_nums and c_nums and not (s1_nums & c_nums):
            continue
        # Rule 2: High name similarity
        ratio = fuzz.token_sort_ratio(s1_name.lower(), c_name.lower())
        if ratio >= 75:
            # If numbers match, lower threshold is fine
            if s1_nums and c_nums and (s1_nums & c_nums):
                pred_ids.add(cid)
            elif ratio >= 85:
                pred_ids.add(cid)

    # evaluate
    tp += len(pred_ids & true_ids)
    fp += len(pred_ids - true_ids)
    fn += len(true_ids - pred_ids)

prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0

print("=" * 60)
print(f"RESULTS ON 5,000 S1 ENTITIES:")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall   : {rec*100:.2f}%")
print(f"  Macro F0.5 (micro proxy): {f05:.4f}")
print("=" * 60)
