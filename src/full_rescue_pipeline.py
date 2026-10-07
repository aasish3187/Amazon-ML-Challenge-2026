"""
Championship Rescue & Dual-Source Completion Pipeline:
Eliminates false singletons and missing source matches across all 3 countries (France, US, India)
using uncapped inverted indexing with strict number/brand consistency.
"""
import os
import sys
import time
import re
from collections import defaultdict
from rapidfuzz import fuzz

sys.path.insert(0, '.')
from src.config import *

STOPWORDS = {
    'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd',
    'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions',
    'enterprises', 'international', 'holdings', 'global', 'technologies',
    'sarl', 'sasu', 'eurl', 'association'
}

def clean_words(n):
    if not n:
        return set()
    return set(re.findall(r'[a-z0-9]{3,}', n.lower())) - STOPWORDS

def extract_nums(a):
    if not a:
        return set()
    nums = re.findall(r'\b\d+\b', a.lower())
    return {n.lstrip('0') for n in nums if n.lstrip('0') and 2 <= len(n) <= 6}

def rescue_country(country: str, current_matches: dict, current_cands: dict):
    print(f"\n{'='*70}")
    print(f"  RESCUING & COMPLETING: {country}")
    print(f"{'='*70}")
    t0 = time.time()
    
    pool_path = f"partitions/test/{country}/pool.tsv"
    s1_path = f"partitions/test/{country}/s1.tsv"
    
    if not os.path.exists(pool_path) or not os.path.exists(s1_path):
        print(f"  Missing partition files for {country}, skipping.")
        return 0, 0

    # 1. Load S1 entities for this country
    s1_records = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            eid = p[0]
            name = p[1] if len(p) > 1 else ''
            addr = p[2] if len(p) > 2 else ''
            s1_records.append((eid, name, addr))

    print(f"  Loaded {len(s1_records):,} S1 records for {country}.")

    # Identify singletons and low-match entities (< 2 matches)
    target_s1 = []
    for sid, name, addr in s1_records:
        m = current_matches.get(sid, [])
        if len(m) < 2:  # singleton or only 1 match
            target_s1.append((sid, name, addr, m))

    print(f"  Targets needing rescue / completion: {len(target_s1):,} ({len(target_s1)/len(s1_records)*100:.1f}%)")

    # 2. Build Uncapped Inverted Index on pool
    print(f"  Building uncapped inverted index on pool...")
    pool = []
    num_to_pool = defaultdict(list)
    word_to_pool = defaultdict(list)
    
    with open(pool_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.strip().split('\t')
            idx = len(pool)
            eid = p[0]
            name = p[1] if len(p) > 1 else ''
            addr = p[2] if len(p) > 2 else ''
            pool.append((eid, name, addr))
            
            for num in extract_nums(addr):
                num_to_pool[num].append(idx)
            for w in clean_words(name):
                word_to_pool[w].append(idx)

    print(f"  Indexed {len(pool):,} pool records in {time.time()-t0:.1f}s.")
    print(f"  Unique numbers: {len(num_to_pool):,}, Unique words: {len(word_to_pool):,}")

    # 3. Query and rescue targets
    t_query = time.time()
    rescued_count = 0
    added_matches_total = 0

    for sid, s1_name, s1_addr, existing_m in target_s1:
        s1_nums = extract_nums(s1_addr)
        s1_words = clean_words(s1_name)
        
        cand_scores = defaultdict(int)
        # Number index
        for num in s1_nums:
            postings = num_to_pool.get(num, [])
            if len(postings) < 2000:
                for pi in postings:
                    cand_scores[pi] += 12
        # Word index
        for w in s1_words:
            postings = word_to_pool.get(w, [])
            if len(postings) < 2000:
                for pi in postings:
                    cand_scores[pi] += 6

        if not cand_scores:
            continue

        existing_set = set(existing_m)
        new_found = []

        for pi, sc in cand_scores.items():
            if sc < 10:
                continue
            cid, c_name, c_addr = pool[pi]
            if cid in existing_set:
                continue
            
            c_nums = extract_nums(c_addr)
            # Rule 1: No conflicting address numbers
            if s1_nums and c_nums and not (s1_nums & c_nums):
                continue

            # Rule 2: Precision Name match
            # If both have matching numbers, threshold = 72
            ratio = fuzz.token_sort_ratio(s1_name.lower(), c_name.lower())
            if s1_nums and c_nums and (s1_nums & c_nums):
                if ratio >= 72:
                    new_found.append((cid, ratio))
            elif ratio >= 85:
                # No numbers: require very high name similarity + common words
                c_words = clean_words(c_name)
                if s1_words & c_words:
                    new_found.append((cid, ratio))

        if new_found:
            # Sort by similarity, keep top matches (max 4 per entity total)
            new_found.sort(key=lambda x: -x[1])
            max_to_add = max(0, 4 - len(existing_m))
            add_ids = [cid for cid, _ in new_found[:max_to_add]]
            
            if add_ids:
                if not existing_m:
                    rescued_count += 1
                updated_m = sorted(list(set(existing_m + add_ids)))
                current_matches[sid] = updated_m
                # Add to candidates to preserve subset rule
                current_cands[sid] = sorted(list(set(current_cands.get(sid, []) + add_ids)))
                added_matches_total += len(add_ids)

    print(f"  Finished {country} in {time.time()-t_query:.1f}s.")
    print(f"  Rescued from singleton: {rescued_count:,}")
    print(f"  Total new high-precision matches added: {added_matches_total:,}")
    
    del pool, num_to_pool, word_to_pool
    import gc
    gc.collect()
    
    return rescued_count, added_matches_total

def main():
    total_t0 = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — CHAMPIONSHIP RESCUE ENGINE")
    print("=" * 70)

    # 1. Load current matches and candidates
    print("Loading current matching_results.tsv and candidate_pairs.tsv...")
    current_matches = {}
    with open('output/matching_results.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            current_matches[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

    current_cands = {}
    with open('output/candidate_pairs.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            current_cands[p[0]] = [x.strip() for x in p[1].split(',') if x.strip()] if len(p) > 1 and p[1].strip() else []

    orig_singletons = sum(1 for m in current_matches.values() if not m)
    print(f"Starting singletons: {orig_singletons:,} ({orig_singletons/len(current_matches)*100:.1f}%)")

    # 2. Rescue per country
    countries = ['France', 'US', 'India']
    total_rescued = 0
    total_added = 0

    for country in countries:
        resc, add = rescue_country(country, current_matches, current_cands)
        total_rescued += resc
        total_added += add

    # 3. Write final updated files
    print("\nWriting updated matching_results.tsv and candidate_pairs.tsv...")
    with open('output/matching_results.tsv', 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f_in:
            next(f_in)
            for line in f_in:
                eid = line.strip().split('\t')[0]
                m = current_matches.get(eid, [])
                f_out.write(f"{eid}\t{','.join(m)}\n")

    with open('output/candidate_pairs.tsv', 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f_in:
            next(f_in)
            for line in f_in:
                eid = line.strip().split('\t')[0]
                c = current_cands.get(eid, [])
                f_out.write(f"{eid}\t{','.join(c)}\n")

    new_singletons = sum(1 for m in current_matches.values() if not m)
    print(f"\n{'='*70}")
    print(f"  🏆 RESCUE COMPLETE in {(time.time()-total_t0)/60:.1f} minutes!")
    print(f"  Singletons reduced: {orig_singletons:,} -> {new_singletons:,} ({new_singletons/len(current_matches)*100:.1f}%)")
    print(f"  Total entities rescued: {total_rescued:,}")
    print(f"  Total high-precision matches added: {total_added:,}")
    print(f"{'='*70}")

    # Verify
    from src.verify_submission import verify
    verify()

if __name__ == '__main__':
    main()
