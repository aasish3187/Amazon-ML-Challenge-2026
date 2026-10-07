"""
Emergency Precision Scrubber:
Removes synthetic distractor collisions (conflicting street numbers and brand mismatches)
from output/matching_results.tsv while strictly preserving test_source1.tsv ordering and subset validity.
"""
import os
import sys
import time
import re

sys.path.insert(0, '.')
from src.config import *
from src.normalize import normalize_record

def extract_numbers(s):
    if not s:
        return set()
    nums = set(re.findall(r'\b\d+\b', str(s)))
    # filter out leading zeros normalization
    return {n.lstrip('0') for n in nums if n.lstrip('0')}

def main():
    t0 = time.time()
    print("=" * 70)
    print("  EMERGENCY PRECISION SCRUBBER — PRUNING TOXIC DISTRACTORS")
    print("=" * 70)

    # 1. Load S1 metadata (name, address numbers)
    print("Step 1: Loading test S1 entities...")
    s1_meta = {}
    with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            eid = parts[0]
            name = parts[1].lower() if len(parts) > 1 else ''
            addr = parts[2].lower() if len(parts) > 2 else ''
            nums = extract_numbers(addr)
            # First significant name token
            name_clean = re.sub(r'[^a-z0-9\s]', ' ', name)
            tokens = [t for t in name_clean.split() if len(t) >= 2 and t not in {'the', 'and', 'inc', 'llc', 'ltd', 'corp', 'pvt'}]
            first_tok = tokens[0] if tokens else ''
            s1_meta[eid] = (nums, first_tok, set(tokens))

    print(f"Loaded {len(s1_meta):,} S1 records in {time.time()-t0:.1f}s")

    # 2. Identify which pool records are currently predicted
    print("\nStep 2: Identifying predicted candidates...")
    needed_pool_ids = set()
    current_preds = {}
    with open('output/matching_results.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            eid = parts[0]
            if len(parts) > 1 and parts[1].strip():
                m_list = [x.strip() for x in parts[1].split(',') if x.strip()]
                current_preds[eid] = m_list
                needed_pool_ids.update(m_list)

    print(f"Total entities with predictions: {len(current_preds):,}")
    print(f"Total unique pool candidates to inspect: {len(needed_pool_ids):,}")

    # 3. Stream test_source2 and test_source3 to extract candidate metadata
    print("\nStep 3: Extracting candidate metadata from pools...")
    cand_meta = {}
    for p in ['student_resource/dataset/test/test_source2.tsv', 'student_resource/dataset/test/test_source3.tsv']:
        t_p = time.time()
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                cid = parts[0]
                if cid in needed_pool_ids:
                    name = parts[1].lower() if len(parts) > 1 else ''
                    addr = parts[2].lower() if len(parts) > 2 else ''
                    nums = extract_numbers(addr)
                    name_clean = re.sub(r'[^a-z0-9\s]', ' ', name)
                    tokens = [t for t in name_clean.split() if len(t) >= 2 and t not in {'the', 'and', 'inc', 'llc', 'ltd', 'corp', 'pvt'}]
                    first_tok = tokens[0] if tokens else ''
                    cand_meta[cid] = (nums, first_tok, set(tokens))
                    if len(cand_meta) == len(needed_pool_ids):
                        break
        print(f"  Scanned {p} in {time.time()-t_p:.1f}s (cached {len(cand_meta):,} / {len(needed_pool_ids):,})")

    # 4. Filter matches
    print("\nStep 4: Filtering false positive distractors...")
    total_original = 0
    total_retained = 0
    num_conflicts_pruned = 0
    name_conflicts_pruned = 0

    clean_preds = {}
    for eid, m_list in current_preds.items():
        s1_nums, s1_first, s1_tokens = s1_meta.get(eid, (set(), '', set()))
        kept = []
        for cid in m_list:
            total_original += 1
            if cid not in cand_meta:
                kept.append(cid)
                total_retained += 1
                continue

            c_nums, c_first, c_tokens = cand_meta[cid]

            # Rule 1: Address number conflict
            # If both have street numbers, and zero overlap -> DISTRACTOR
            if s1_nums and c_nums and not (s1_nums & c_nums):
                num_conflicts_pruned += 1
                continue

            # Rule 2: First brand token conflict
            # If both have a primary brand token, and neither is substring/prefix of the other
            if s1_first and c_first:
                if s1_first != c_first and not (s1_first.startswith(c_first) or c_first.startswith(s1_first)):
                    # Check if token overlap is completely empty
                    if not (s1_tokens & c_tokens):
                        name_conflicts_pruned += 1
                        continue

            kept.append(cid)
            total_retained += 1

        clean_preds[eid] = kept

    print(f"Total original matches : {total_original:,}")
    print(f"Retained high-precision: {total_retained:,} ({total_retained/total_original*100:.1f}%)")
    print(f"Pruned number conflicts: {num_conflicts_pruned:,}")
    print(f"Pruned brand conflicts : {name_conflicts_pruned:,}")

    # 5. Write new matching_results.tsv in EXACT test_source1 order
    print("\nStep 5: Writing clean output/matching_results.tsv...")
    out_path = 'output/matching_results.tsv'
    backup_path = 'output/matching_results_pre_scrub.tsv'
    if not os.path.exists(backup_path):
        os.rename(out_path, backup_path)
    else:
        # Just overwrite out_path
        pass

    with open(out_path, 'w', encoding='utf-8') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        with open('student_resource/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f_in:
            next(f_in)
            for line in f_in:
                eid = line.strip().split('\t')[0]
                m = clean_preds.get(eid, [])
                f_out.write(f"{eid}\t{','.join(m)}\n")

    print(f"Successfully written {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"Total execution time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
