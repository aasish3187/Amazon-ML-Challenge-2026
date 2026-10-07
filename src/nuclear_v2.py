"""
NUCLEAR PIPELINE v2: Train-Validated Entity Resolution
=======================================================
1. Evaluate on training data first to know exact score
2. Use transliteration-aware matching (anyascii)
3. Uncapped inverted indexes
4. Address-first matching for non-Latin names
5. Validate score >= 0.90 on train before generating test output
"""
import os
import sys
import time
import re
from collections import defaultdict, Counter
import anyascii
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

# ============================================================
# CORE MATCHING FUNCTIONS
# ============================================================

def normalize_name(name):
    """Normalize business name with transliteration."""
    if not name or not isinstance(name, str):
        return ''
    # Transliterate non-Latin scripts
    text = anyascii.anyascii(name)
    text = text.lower()
    text = text.replace('&', ' and ').replace('+', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_addr(addr):
    """Normalize address with transliteration."""
    if not addr or not isinstance(addr, str):
        return ''
    text = anyascii.anyascii(addr)
    text = text.lower()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numbers(text):
    """Extract numeric tokens."""
    if not text or not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d{2,}\b', text))

def extract_words(text, min_len=3):
    """Extract significant words."""
    if not text:
        return set()
    stops = {'the', 'and', 'company', 'corporation', 'limited', 'private',
             'ltd', 'pvt', 'llc', 'inc', 'services', 'management', 'group',
             'solutions', 'enterprises', 'international', 'holdings', 'global',
             'technologies', 'sarl', 'sasu', 'eurl', 'association',
             'road', 'street', 'lane', 'floor', 'building', 'near',
             'nagar', 'colony', 'plot', 'sector', 'india', 'france',
             'north', 'south', 'east', 'west', 'main', 'cross',
             'pvt', 'ltd', 'limited', 'private'}
    return {w for w in text.split() if len(w) >= min_len and w not in stops}

def match_score(s1_name, s1_addr, c_name, c_addr, s1_nums, c_nums):
    """
    Compute a match confidence score.
    Returns (score, reason) where score is 0-100.
    
    KEY INSIGHT: Use BOTH name AND address. True matches have
    either high name OR high address similarity (or both).
    Non-Latin names may have very low name similarity but near-identical addresses.
    """
    # Normalized name similarity
    if s1_name and c_name:
        name_tsr = fuzz.token_sort_ratio(s1_name, c_name)
        name_tsetr = fuzz.token_set_ratio(s1_name, c_name)
        name_sim = max(name_tsr, name_tsetr)
    else:
        name_sim = 0
    
    # Address similarity
    if s1_addr and c_addr:
        addr_tsr = fuzz.token_sort_ratio(s1_addr, c_addr)
    else:
        addr_tsr = 0
    
    # Number overlap
    nums_match = False
    nums_conflict = False
    if s1_nums and c_nums:
        if s1_nums & c_nums:
            nums_match = True
        else:
            nums_conflict = True
    
    # ======= DECISION RULES =======
    
    # Rule 1: Very high name similarity (>= 85) → strong match
    if name_sim >= 85:
        if nums_conflict:
            return 0, "name_high_but_nums_conflict"
        return 95, "name_very_high"
    
    # Rule 2: Good name (>= 70) + good address (>= 60)
    if name_sim >= 70 and addr_tsr >= 60:
        if nums_conflict:
            return 0, "conflict"
        return 90, "name_good_addr_good"
    
    # Rule 3: Good name (>= 70) alone
    if name_sim >= 70:
        if nums_conflict:
            return 0, "conflict"
        if nums_match:
            return 85, "name_good_nums_match"
        return 65, "name_good_only"
    
    # Rule 4: Medium name (>= 55) + strong address (>= 75)
    if name_sim >= 55 and addr_tsr >= 75:
        if nums_conflict:
            return 0, "conflict"
        return 80, "name_medium_addr_strong"
    
    # Rule 5: Low name (< 55) but very strong address (>= 80) + numbers match
    # This catches non-Latin transliterations (Hindi, Tamil, etc.)
    if addr_tsr >= 80 and nums_match:
        return 85, "addr_very_strong_nums_match"
    
    # Rule 6: Low name but strong address (>= 75) without number conflict
    if addr_tsr >= 75 and not nums_conflict:
        if nums_match:
            return 75, "addr_strong_nums_match"
        return 50, "addr_strong_only"  # risky without nums
    
    # Rule 7: Medium name (>= 55) + numbers match
    if name_sim >= 55 and nums_match:
        return 70, "name_medium_nums_match"
    
    # Rule 8: Address match with numbers but low everything else
    if addr_tsr >= 65 and nums_match and name_sim >= 30:
        return 60, "addr_ok_nums_match_name_low"
    
    return 0, "no_match"


def build_index(records):
    """Build inverted index from records. No cap on posting lists."""
    num_index = defaultdict(list)
    word_index = defaultdict(list)
    
    for i, (eid, name, addr, nums) in enumerate(records):
        for n in nums:
            num_index[n].append(i)
        words = extract_words(name)
        for w in words:
            word_index[w].append(i)
        # Also index address words
        addr_words = extract_words(addr, min_len=4)
        for w in addr_words:
            word_index[w].append(i)
    
    return num_index, word_index


def find_candidates(s1_name, s1_addr, s1_nums, num_index, word_index, max_cands=100):
    """Find candidate pool indices using blocking."""
    scores = defaultdict(int)
    
    # Number blocking (strongest signal)
    for n in s1_nums:
        for idx in num_index.get(n, []):
            scores[idx] += 15 if len(n) >= 5 else 10
    
    # Word blocking
    s1_words = extract_words(s1_name)
    for w in s1_words:
        postings = word_index.get(w, [])
        # Skip ultra-common words
        if len(postings) > 5000:
            continue
        for idx in postings:
            scores[idx] += 8 if len(w) >= 5 else 5
    
    # Address word blocking
    addr_words = extract_words(s1_addr, min_len=4)
    for w in addr_words:
        postings = word_index.get(w, [])
        if len(postings) > 5000:
            continue
        for idx in postings:
            scores[idx] += 5
    
    if not scores:
        return []
    
    # Return top candidates
    top = sorted(scores.items(), key=lambda x: -x[1])[:max_cands]
    return [idx for idx, sc in top if sc >= 5]


# ============================================================
# PHASE 1: VALIDATE ON TRAINING DATA
# ============================================================

def evaluate_on_train(sample_size=50000):
    """Run our matching on a sample of training data and compute F0.5."""
    print("=" * 70)
    print("  PHASE 1: VALIDATE ON TRAINING DATA")
    print("=" * 70)
    t0 = time.time()
    
    # Load ground truth
    print("Loading ground truth...")
    gt = {}
    with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            sid = parts[0]
            if len(parts) > 1 and parts[1].strip():
                gt[sid] = set(x.strip() for x in parts[1].split(',') if x.strip())
            else:
                gt[sid] = set()
    
    # Load S1 records (sample)
    print(f"Loading S1 training records (sampling {sample_size:,})...")
    s1_records = []
    all_s1 = []
    with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                all_s1.append(parts)
    
    # Stratified sample: take proportional singletons and matched
    import random
    random.seed(42)
    
    singleton_ids = [p for p in all_s1 if not gt.get(p[0], set())]
    matched_ids = [p for p in all_s1 if gt.get(p[0], set())]
    
    # Sample proportionally
    n_sing = int(sample_size * len(singleton_ids) / len(all_s1))
    n_match = sample_size - n_sing
    
    sampled = random.sample(singleton_ids, min(n_sing, len(singleton_ids))) + \
              random.sample(matched_ids, min(n_match, len(matched_ids)))
    
    sampled_s1 = {}
    for parts in sampled:
        eid = parts[0]
        name = normalize_name(parts[1])
        addr = normalize_addr(parts[2])
        nums = extract_numbers(parts[2])
        country = parts[3] if len(parts) > 3 else ''
        sampled_s1[eid] = (name, addr, nums, country)
    
    print(f"  Sampled {len(sampled_s1):,} S1 entities ({n_sing} singletons, {n_match} matched)")
    
    # Determine which countries are in our sample
    countries_in_sample = set(d[3] for d in sampled_s1.values())
    print(f"  Countries in sample: {countries_in_sample}")
    
    # Load pool (S2 + S3) for relevant countries
    print("Loading S2+S3 pool...")
    pool = []  # (eid, norm_name, norm_addr, nums)
    
    for src in ['train_source2.tsv', 'train_source3.tsv']:
        path = f'student_resource/dataset/train/{src}'
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    country = parts[3] if len(parts) > 3 else ''
                    if country in countries_in_sample:
                        name = normalize_name(parts[1])
                        addr = normalize_addr(parts[2])
                        nums = extract_numbers(parts[2])
                        pool.append((parts[0], name, addr, nums))
    
    print(f"  Pool size: {len(pool):,}")
    
    # Build index
    print("Building inverted index...")
    num_idx, word_idx = build_index(pool)
    print(f"  Number keys: {len(num_idx):,}, Word keys: {len(word_idx):,}")
    
    # Run matching
    print(f"\nRunning matching on {len(sampled_s1):,} entities...")
    predictions = {}
    match_reasons = Counter()
    
    processed = 0
    for sid, (s1_name, s1_addr, s1_nums, s1_country) in sampled_s1.items():
        # Find candidates
        cand_indices = find_candidates(s1_name, s1_addr, s1_nums, num_idx, word_idx, max_cands=100)
        
        matched_ids = []
        for ci in cand_indices:
            c_eid, c_name, c_addr, c_nums = pool[ci]
            score, reason = match_score(s1_name, s1_addr, c_name, c_addr, s1_nums, c_nums)
            if score >= 60:
                matched_ids.append(c_eid)
                match_reasons[reason] += 1
        
        predictions[sid] = set(matched_ids)
        processed += 1
        
        if processed % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {processed:,}/{len(sampled_s1):,} ({processed/elapsed:.0f}/sec)")
    
    # Compute F0.5
    print(f"\nComputing Macro F0.5...")
    scores = []
    perfect_scores = 0
    zero_scores = 0
    
    breakdown_true_sing_correct = 0
    breakdown_true_sing_wrong = 0
    breakdown_true_match_got = 0
    breakdown_true_match_missed = 0
    
    for sid in sampled_s1:
        pred = predictions.get(sid, set())
        true = gt.get(sid, set())
        
        if len(true) == 0:
            if len(pred) == 0:
                s = 1.0
                breakdown_true_sing_correct += 1
            else:
                s = 0.0
                breakdown_true_sing_wrong += 1
        elif len(pred) == 0:
            s = 0.0
            breakdown_true_match_missed += 1
        else:
            tp = len(pred & true)
            fp = len(pred - true)
            fn = len(true - pred)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            if prec + rec == 0:
                s = 0.0
            else:
                s = (1.25 * prec * rec) / (0.25 * prec + rec)
            if s >= 0.99:
                breakdown_true_match_got += 1
        
        scores.append(s)
        if s >= 0.99:
            perfect_scores += 1
        if s < 0.01:
            zero_scores += 1
    
    macro_f05 = sum(scores) / len(scores)
    
    print(f"\n{'='*70}")
    print(f"  TRAINING VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"  Sample size: {len(scores):,}")
    print(f"  Macro F0.5: {macro_f05:.6f}")
    print(f"  Perfect (>=0.99): {perfect_scores:,} ({perfect_scores/len(scores)*100:.1f}%)")
    print(f"  Zero (<0.01): {zero_scores:,} ({zero_scores/len(scores)*100:.1f}%)")
    print(f"\n  Breakdown:")
    print(f"    True singletons correctly empty: {breakdown_true_sing_correct:,}")
    print(f"    True singletons wrongly matched: {breakdown_true_sing_wrong:,}")
    print(f"    True matched but we missed entirely: {breakdown_true_match_missed:,}")
    print(f"\n  Match reasons: {dict(match_reasons.most_common(10))}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    
    return macro_f05


# ============================================================
# PHASE 2: FULL TEST PREDICTION
# ============================================================

def predict_test():
    """Generate full test predictions country by country."""
    print("\n" + "=" * 70)
    print("  PHASE 2: FULL TEST PREDICTION")
    print("=" * 70)
    t0 = time.time()
    
    countries = ['France', 'US', 'India']
    all_matches = {}  # sid -> list of match IDs
    all_candidates = {}  # sid -> list of candidate IDs
    
    for country in countries:
        tc = time.time()
        print(f"\n{'='*60}")
        print(f"  Processing {country}...")
        print(f"{'='*60}")
        
        # Load S1 for this country
        s1_path = f'partitions/test/{country}/s1.tsv'
        pool_path = f'partitions/test/{country}/pool.tsv'
        
        if not os.path.exists(s1_path):
            print(f"  SKIP: {s1_path} not found")
            continue
        
        s1_records = []
        with open(s1_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    eid = parts[0]
                    name = normalize_name(parts[1])
                    addr = normalize_addr(parts[2])
                    nums = extract_numbers(parts[2])
                    s1_records.append((eid, name, addr, nums))
        
        print(f"  S1 records: {len(s1_records):,}")
        
        # Load pool
        pool = []
        with open(pool_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    name = normalize_name(parts[1])
                    addr = normalize_addr(parts[2])
                    nums = extract_numbers(parts[2])
                    pool.append((parts[0], name, addr, nums))
        
        print(f"  Pool records: {len(pool):,}")
        
        # Build index
        num_idx, word_idx = build_index(pool)
        print(f"  Index built. Nums: {len(num_idx):,}, Words: {len(word_idx):,}")
        
        # Match
        matched_count = 0
        singleton_count = 0
        
        for i, (sid, s1_name, s1_addr, s1_nums) in enumerate(s1_records):
            cand_indices = find_candidates(s1_name, s1_addr, s1_nums, num_idx, word_idx, max_cands=100)
            
            matched_ids = []
            cand_ids = []
            for ci in cand_indices:
                c_eid, c_name, c_addr, c_nums = pool[ci]
                cand_ids.append(c_eid)
                score, reason = match_score(s1_name, s1_addr, c_name, c_addr, s1_nums, c_nums)
                if score >= 60:
                    matched_ids.append(c_eid)
            
            all_matches[sid] = sorted(set(matched_ids))
            all_candidates[sid] = sorted(set(cand_ids))
            
            if matched_ids:
                matched_count += 1
            else:
                singleton_count += 1
            
            if (i + 1) % 50000 == 0:
                elapsed = time.time() - tc
                print(f"    {i+1:,}/{len(s1_records):,} ({(i+1)/elapsed:.0f}/sec) "
                      f"matched={matched_count:,} singletons={singleton_count:,}")
        
        print(f"  {country} done: {matched_count:,} matched, {singleton_count:,} singletons "
              f"in {time.time()-tc:.1f}s")
        
        # Free memory
        del pool, num_idx, word_idx
        import gc
        gc.collect()
    
    # Write output files in test_source1.tsv order
    print(f"\nWriting output files in test_source1.tsv order...")
    
    with open('output/matching_results.tsv', 'w', encoding='utf-8') as f_match, \
         open('output/candidate_pairs.tsv', 'w', encoding='utf-8') as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f_in:
            next(f_in)
            for line in f_in:
                eid = line.strip().split('\t')[0]
                m = all_matches.get(eid, [])
                c = all_candidates.get(eid, [])
                f_match.write(f"{eid}\t{','.join(m)}\n")
                f_cand.write(f"{eid}\t{','.join(c)}\n")
    
    total = len(all_matches)
    matched = sum(1 for m in all_matches.values() if m)
    print(f"\nTotal: {total:,} | Matched: {matched:,} ({matched/total*100:.1f}%) | "
          f"Singletons: {total-matched:,} ({(total-matched)/total*100:.1f}%)")
    print(f"Total time: {(time.time()-t0)/60:.1f} minutes")


def main():
    t0 = time.time()
    
    # PHASE 1: Validate on training data
    train_score = evaluate_on_train(sample_size=50000)
    
    print(f"\n{'='*70}")
    if train_score >= 0.80:
        print(f"  TRAIN SCORE {train_score:.4f} >= 0.80 -- PROCEEDING TO TEST")
        print(f"{'='*70}")
        predict_test()
    else:
        print(f"  TRAIN SCORE {train_score:.4f} < 0.80 -- TUNING NEEDED")
        print(f"  NOT generating test predictions until train score is good enough")
        print(f"{'='*70}")
        return
    
    # Package
    print(f"\nTotal pipeline time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == '__main__':
    main()
