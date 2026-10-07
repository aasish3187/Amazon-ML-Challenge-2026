"""
NUCLEAR PIPELINE v3: Fast Train-Validated Entity Resolution
=============================================================
Key optimizations:
- Cap posting lists at 10,000 (vs 3,000 old / uncapped v2)
- Smaller validation sample (10k) but enough to be statistically valid
- Process candidates in batches with early termination
- Skip ultra-common postings entirely
"""
import os
import sys
import time
import re
from collections import defaultdict, Counter
import anyascii
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

MAX_POSTING = 10000  # Much larger than 3000 but not uncapped

def norm_name(name):
    if not name or not isinstance(name, str): return ''
    t = anyascii.anyascii(name).lower()
    t = t.replace('&', ' and ').replace('+', ' and ')
    t = re.sub(r'[^\w\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def norm_addr(addr):
    if not addr or not isinstance(addr, str): return ''
    t = anyascii.anyascii(addr).lower()
    t = t.replace('&', ' and ')
    t = re.sub(r'[^\w\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def get_nums(text):
    if not text: return set()
    return set(re.findall(r'\b\d{2,}\b', str(text)))

STOPS = {'the','and','company','corporation','limited','private','ltd','pvt',
         'llc','inc','services','management','group','solutions','enterprises',
         'international','holdings','global','technologies','sarl','sasu','eurl',
         'association','road','street','lane','floor','building','near','nagar',
         'colony','plot','sector','india','france','north','south','east','west',
         'main','cross','door','shop','flat','apartment','behind','opposite',
         'opp','hno','dno','new','old'}

def get_words(text, min_len=3):
    if not text: return set()
    return {w for w in text.split() if len(w) >= min_len and w not in STOPS}


def score_pair(n1, a1, nums1, n2, a2, nums2):
    """Score a candidate pair. Returns 0-100."""
    # Name similarity (transliteration-aware since both are already anyascii'd)
    if n1 and n2:
        nsim = max(fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2))
    else:
        nsim = 0
    
    # Address similarity
    asim = fuzz.token_sort_ratio(a1, a2) if a1 and a2 else 0
    
    # Number analysis
    num_match = bool(nums1 and nums2 and (nums1 & nums2))
    num_conflict = bool(nums1 and nums2 and not (nums1 & nums2))
    
    # Hard reject: conflicting numbers with low name
    if num_conflict and nsim < 85:
        return 0
    
    # === MATCHING RULES (ordered by confidence) ===
    
    # R1: Very high name (>=85) → match (unless nums conflict)
    if nsim >= 85:
        return 95
    
    # R2: Good name (>=70) + good addr (>=60)
    if nsim >= 70 and asim >= 60:
        return 90
    
    # R3: Good name (>=70) + nums match
    if nsim >= 70 and num_match:
        return 88
    
    # R4: Good name (>=70) alone — moderate confidence
    if nsim >= 70:
        return 68
    
    # R5: Medium name (>=55) + strong addr (>=70) 
    if nsim >= 55 and asim >= 70:
        return 82
    
    # R6: Medium name (>=55) + nums match
    if nsim >= 55 and num_match:
        return 75
    
    # R7: LOW name but VERY strong addr (>=80) + nums match
    # KEY: catches non-Latin transliteration matches (Hindi/Tamil → Latin)
    if asim >= 80 and num_match:
        return 85
    
    # R8: LOW name but strong addr (>=70) + nums match  
    if asim >= 70 and num_match and nsim >= 20:
        return 70
    
    # R9: Very strong addr (>=85) alone (no nums to verify)
    if asim >= 85 and nsim >= 30:
        return 65
    
    return 0


def load_pool(path):
    """Load pool records from TSV."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 3:
                eid = p[0]
                n = norm_name(p[1])
                a = norm_addr(p[2])
                nums = get_nums(p[2])
                records.append((eid, n, a, nums))
    return records


def build_index(records):
    """Build inverted index with capped posting lists."""
    num_idx = defaultdict(list)
    word_idx = defaultdict(list)
    
    for i, (eid, name, addr, nums) in enumerate(records):
        for n in nums:
            if len(num_idx[n]) < MAX_POSTING:
                num_idx[n].append(i)
        for w in get_words(name):
            if len(word_idx[w]) < MAX_POSTING:
                word_idx[w].append(i)
        for w in get_words(addr, min_len=5):
            if len(word_idx[w]) < MAX_POSTING:
                word_idx[w].append(i)
    
    return num_idx, word_idx


def find_matches(s1_name, s1_addr, s1_nums, pool, num_idx, word_idx, 
                 max_cands=80, threshold=60):
    """Find matching pool records for an S1 entity."""
    # Blocking: score candidates
    cand_scores = defaultdict(int)
    
    for n in s1_nums:
        for idx in num_idx.get(n, []):
            cand_scores[idx] += 12 if len(n) >= 5 else 8
    
    for w in get_words(s1_name):
        postings = word_idx.get(w, [])
        if len(postings) > 5000:
            continue  # skip ultra-common
        weight = 8 if len(w) >= 5 else 5
        for idx in postings:
            cand_scores[idx] += weight
    
    for w in get_words(s1_addr, min_len=5):
        postings = word_idx.get(w, [])
        if len(postings) > 5000:
            continue
        for idx in postings:
            cand_scores[idx] += 4
    
    if not cand_scores:
        return [], []
    
    # Top candidates
    top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:max_cands]
    
    matched = []
    all_cands = []
    
    for idx, blocking_score in top_cands:
        if blocking_score < 4:
            continue
        c_eid, c_name, c_addr, c_nums = pool[idx]
        all_cands.append(c_eid)
        
        sc = score_pair(s1_name, s1_addr, s1_nums, c_name, c_addr, c_nums)
        if sc >= threshold:
            matched.append(c_eid)
    
    return sorted(set(matched)), sorted(set(all_cands))


def validate_on_train():
    """Validate on training data."""
    print("=" * 70)
    print("  PHASE 1: TRAINING VALIDATION (10k sample)")
    print("=" * 70)
    t0 = time.time()
    
    # Load GT
    gt = {}
    with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            gt[p[0]] = set(x.strip() for x in p[1].split(',') if x.strip()) if len(p) > 1 and p[1].strip() else set()
    
    # Load S1 sample
    import random
    random.seed(42)
    all_s1 = []
    with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                all_s1.append(p)
    
    sample = random.sample(all_s1, min(10000, len(all_s1)))
    countries = set(p[3] for p in sample if len(p) > 3)
    print(f"  Sample: {len(sample):,} S1 entities, countries: {countries}")
    
    # Load pool
    pool = []
    for src in ['train_source2.tsv', 'train_source3.tsv']:
        with open(f'student_resource/dataset/train/{src}', 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 3:
                    pool.append((p[0], norm_name(p[1]), norm_addr(p[2]), get_nums(p[2])))
    
    print(f"  Pool: {len(pool):,} records. Building index...")
    num_idx, word_idx = build_index(pool)
    print(f"  Index: {len(num_idx):,} nums, {len(word_idx):,} words")
    
    # Match
    print(f"  Matching...")
    scores = []
    false_singletons = 0
    false_matches_on_singletons = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, parts in enumerate(sample):
        sid = parts[0]
        n = norm_name(parts[1])
        a = norm_addr(parts[2])
        nums = get_nums(parts[2])
        
        matched, _ = find_matches(n, a, nums, pool, num_idx, word_idx, threshold=60)
        pred = set(matched)
        true = gt.get(sid, set())
        
        if len(true) == 0:
            s = 1.0 if len(pred) == 0 else 0.0
            if len(pred) > 0:
                false_matches_on_singletons += 1
        elif len(pred) == 0:
            s = 0.0
            false_singletons += 1
        else:
            tp = len(pred & true)
            fp = len(pred - true)
            fn = len(true - pred)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            s = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        
        scores.append(s)
        
        if (i+1) % 1000 == 0:
            cur_f05 = sum(scores) / len(scores)
            print(f"    {i+1:,}/10k  Running F0.5={cur_f05:.4f}  "
                  f"false_sing={false_singletons} false_match_on_sing={false_matches_on_singletons}")
    
    macro = sum(scores) / len(scores)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"  TRAINING VALIDATION RESULT")
    print(f"{'='*70}")
    print(f"  Macro F0.5:              {macro:.6f}")
    print(f"  Micro Precision:         {micro_prec:.4f}")
    print(f"  Micro Recall:            {micro_rec:.4f}")
    print(f"  False singletons:        {false_singletons:,} (true matched but we returned empty)")
    print(f"  False matches on sing:   {false_matches_on_singletons:,} (true singleton but we matched)")
    print(f"  Perfect scores (>=0.99): {sum(1 for s in scores if s >= 0.99):,}")
    print(f"  Zero scores:             {sum(1 for s in scores if s < 0.01):,}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    return macro


def predict_test():
    """Full test prediction."""
    print("\n" + "=" * 70)
    print("  PHASE 2: TEST PREDICTION")
    print("=" * 70)
    t0 = time.time()
    
    all_matches = {}
    all_cands = {}
    
    for country in ['France', 'US', 'India']:
        tc = time.time()
        s1_path = f'partitions/test/{country}/s1.tsv'
        pool_path = f'partitions/test/{country}/pool.tsv'
        
        if not os.path.exists(s1_path):
            print(f"  SKIP {country}")
            continue
        
        print(f"\n  === {country} ===")
        
        # Load pool
        pool = load_pool(pool_path)
        print(f"  Pool: {len(pool):,}")
        
        num_idx, word_idx = build_index(pool)
        print(f"  Index: {len(num_idx):,} nums, {len(word_idx):,} words")
        
        # Load and process S1
        s1_count = 0
        matched_count = 0
        
        with open(s1_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                p = line.strip().split('\t')
                if len(p) < 3:
                    continue
                
                sid = p[0]
                n = norm_name(p[1])
                a = norm_addr(p[2])
                nums = get_nums(p[2])
                
                m, c = find_matches(n, a, nums, pool, num_idx, word_idx, threshold=60)
                all_matches[sid] = m
                all_cands[sid] = c
                
                s1_count += 1
                if m:
                    matched_count += 1
                
                if s1_count % 50000 == 0:
                    elapsed = time.time() - tc
                    print(f"    {s1_count:,} ({s1_count/elapsed:.0f}/sec) "
                          f"matched={matched_count:,} sing={s1_count-matched_count:,} "
                          f"({(s1_count-matched_count)/s1_count*100:.1f}%)")
        
        print(f"  {country}: {s1_count:,} S1, {matched_count:,} matched "
              f"({matched_count/s1_count*100:.1f}%), {time.time()-tc:.1f}s")
        
        del pool, num_idx, word_idx
        import gc; gc.collect()
    
    # Write in test_source1 order
    print(f"\n  Writing output files...")
    with open('output/matching_results.tsv', 'w', encoding='utf-8') as fm, \
         open('output/candidate_pairs.tsv', 'w', encoding='utf-8') as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        
        with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as fi:
            next(fi)
            for line in fi:
                eid = line.strip().split('\t')[0]
                m = all_matches.get(eid, [])
                c = all_cands.get(eid, [])
                fm.write(f"{eid}\t{','.join(m)}\n")
                fc.write(f"{eid}\t{','.join(c)}\n")
    
    total = len(all_matches)
    matched = sum(1 for m in all_matches.values() if m)
    print(f"\n  Total: {total:,} | Matched: {matched:,} ({matched/total*100:.1f}%) | "
          f"Singletons: {total-matched:,} ({(total-matched)/total*100:.1f}%)")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")


def main():
    t0 = time.time()
    
    train_score = validate_on_train()
    
    if train_score >= 0.60:
        print(f"\n  Train score {train_score:.4f} -- proceeding to test predictions")
        predict_test()
        
        # Package
        print(f"\n  Packaging submission...")
        import zipfile
        with zipfile.ZipFile('output/submission.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write('output/matching_results.tsv', 'matching_results.tsv')
            zf.write('output/candidate_pairs.tsv', 'candidate_pairs.tsv')
        sz = os.path.getsize('output/submission.zip') / 1e6
        print(f"  submission.zip: {sz:.1f} MB")
    else:
        print(f"\n  Train score {train_score:.4f} too low -- aborting")
    
    print(f"\n  TOTAL TIME: {(time.time()-t0)/60:.1f} minutes")


if __name__ == '__main__':
    main()
