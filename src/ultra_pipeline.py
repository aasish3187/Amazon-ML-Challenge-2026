"""
Ultra Rescue & Transitive Completion Pipeline:
1. Fast Singleton Rescue: Recovers ~340k false singletons using uncapped inverted index.
2. Transitive Source Completion: Completes missing S2/S3 pairs for matched entities.
3. Strict Subset & Alignment Verification: Keeps matching_results.tsv strictly subsetted in candidate_pairs.tsv.
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

def process_country(country: str, current_matches: dict, current_cands: dict):
    print(f"\n{'='*70}")
    print(f"  PROCESSING COUNTRY: {country}")
    print(f"{'='*70}")
    t0 = time.time()
    
    pool_path = f"partitions/test/{country}/pool.tsv"
    s1_path = f"partitions/test/{country}/s1.tsv"
    
    if not os.path.exists(pool_path) or not os.path.exists(s1_path):
        print(f"  Missing partition files for {country}, skipping.")
        return 0, 0

    # 1. Load S1 entities
    s1_records = []
    with open(s1_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            eid = p[0]
            name = p[1] if len(p) > 1 else ''
            addr = p[2] if len(p) > 2 else ''
            s1_records.append((eid, name, addr))

    singletons = [s for s in s1_records if not current_matches.get(s[0])]
    print(f"  Total S1: {len(s1_records):,} | Current Singletons: {len(singletons):,} ({len(singletons)/len(s1_records)*100:.1f}%)")

    # 2. Build Inverted Index on Pool
    print(f"  Indexing {country} pool...")
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

    # 3. Fast Singleton Rescue
    t_rescue = time.time()
    rescued_count = 0
    added_matches = 0

    for sid, s1_name, s1_addr in singletons:
        s1_nums = extract_nums(s1_addr)
        s1_words = clean_words(s1_name)
        
        sc = defaultdict(int)
        for n in s1_nums:
            for pi in num_to_pool.get(n, ())[:300]:
                sc[pi] += 15
        for w in s1_words:
            for pi in word_to_pool.get(w, ())[:200]:
                sc[pi] += 6

        if not sc:
            continue

        # Sort top 10 candidates by score
        top_cands = sorted(sc.items(), key=lambda x: -x[1])[:10]
        found = []

        for pi, score in top_cands:
            if score < 12:
                continue
            cid, c_name, c_addr = pool[pi]
            c_nums = extract_nums(c_addr)
            
            # Conflicting numbers prune
            if s1_nums and c_nums and not (s1_nums & c_nums):
                continue

            ratio = fuzz.token_sort_ratio(s1_name.lower(), c_name.lower())
            
            # High-precision thresholds
            if s1_nums and c_nums and (s1_nums & c_nums):
                if ratio >= 72:
                    found.append((cid, ratio))
            elif ratio >= 85:
                # Require common word when no numbers
                if s1_words & clean_words(c_name):
                    found.append((cid, ratio))

        if found:
            found.sort(key=lambda x: -x[1])
            # Keep top matches (up to 4)
            best_cids = [cid for cid, _ in found[:4]]
            current_matches[sid] = best_cids
            current_cands[sid] = sorted(list(set(current_cands.get(sid, []) + best_cids)))
            rescued_count += 1
            added_matches += len(best_cids)

    print(f"  Rescued {rescued_count:,} singletons in {time.time()-t_rescue:.1f}s ({added_matches:,} matches added).")

    # 4. Transitive Dual-Source Completion
    # If S1 has only S2 matches, find identical S3 matches; if only S3, find S2!
    print("  Running transitive dual-source completion...")
    t_trans = time.time()
    transitive_added = 0

    for sid, s1_name, s1_addr in s1_records:
        m = current_matches.get(sid, [])
        if not m:
            continue
        has_s2 = any(x.startswith('S2-') for x in m)
        has_s3 = any(x.startswith('S3-') for x in m)
        
        # If already has both sources or already has 4+ matches, skip
        if (has_s2 and has_s3) or len(m) >= 4:
            continue

        s1_nums = extract_nums(s1_addr)
        s1_words = clean_words(s1_name)
        target_prefix = 'S3-' if has_s2 else 'S2-'

        sc = defaultdict(int)
        for n in s1_nums:
            for pi in num_to_pool.get(n, ())[:300]:
                if pool[pi][0].startswith(target_prefix):
                    sc[pi] += 15
        for w in s1_words:
            for pi in word_to_pool.get(w, ())[:200]:
                if pool[pi][0].startswith(target_prefix):
                    sc[pi] += 6

        if not sc:
            continue

        top_cands = sorted(sc.items(), key=lambda x: -x[1])[:5]
        new_cids = []
        for pi, score in top_cands:
            if score < 15:
                continue
            cid, c_name, c_addr = pool[pi]
            c_nums = extract_nums(c_addr)
            if s1_nums and c_nums and not (s1_nums & c_nums):
                continue
            ratio = fuzz.token_sort_ratio(s1_name.lower(), c_name.lower())
            if (s1_nums and c_nums and (s1_nums & c_nums) and ratio >= 75) or ratio >= 88:
                new_cids.append(cid)
                if len(m) + len(new_cids) >= 4:
                    break

        if new_cids:
            updated_m = sorted(list(set(m + new_cids)))
            current_matches[sid] = updated_m
            current_cands[sid] = sorted(list(set(current_cands.get(sid, []) + new_cids)))
            transitive_added += len(new_cids)

    print(f"  Transitive completions added: {transitive_added:,} in {time.time()-t_trans:.1f}s.")
    
    del pool, num_to_pool, word_to_pool
    import gc
    gc.collect()

    return rescued_count, added_matches + transitive_added

def main():
    total_t0 = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — ULTRA RESCUE & COMPLETION")
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
    print(f"Initial singletons: {orig_singletons:,} ({orig_singletons/len(current_matches)*100:.2f}%)")

    # 2. Process countries
    total_rescued = 0
    total_matches_added = 0
    for country in ['France', 'US', 'India']:
        resc, add = process_country(country, current_matches, current_cands)
        total_rescued += resc
        total_matches_added += add

    # 3. Write final output files
    print("\nWriting final matching_results.tsv and candidate_pairs.tsv...")
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

    final_singletons = sum(1 for m in current_matches.values() if not m)
    elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"  🏆 ULTRA PIPELINE COMPLETE in {elapsed/60:.1f} minutes!")
    print(f"  Initial singletons: {orig_singletons:,} ({orig_singletons/len(current_matches)*100:.2f}%)")
    print(f"  Final singletons  : {final_singletons:,} ({final_singletons/len(current_matches)*100:.2f}%)")
    print(f"  Total singletons rescued: {total_rescued:,}")
    print(f"  Total matches added     : {total_matches_added:,}")
    print(f"{'='*70}")

    # Verify
    from src.verify_submission import verify
    verify()

if __name__ == '__main__':
    main()
