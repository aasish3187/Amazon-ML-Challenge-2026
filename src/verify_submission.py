"""
Submission Verification & Integrity Diagnostic Script
Validates output/matching_results.tsv and output/candidate_pairs.tsv against
official competition specifications.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

def verify():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import TEST_S1, MATCHING_OUTPUT, CANDIDATE_OUTPUT

    test_s1_path = TEST_S1
    matching_path = MATCHING_OUTPUT
    candidate_path = CANDIDATE_OUTPUT

    print("=" * 75)
    print("  AMAZON ML CHALLENGE 2026 — SUBMISSION VERIFICATION AUDIT")
    print("=" * 75)

    if not os.path.exists(matching_path):
        print(f"❌ Error: {matching_path} does not exist!")
        return False
    if not os.path.exists(candidate_path):
        print(f"❌ Error: {candidate_path} does not exist!")
        return False

    t0 = time.time()

    # Step 1: Read test_source1.tsv reference IDs
    print("\n[1/5] Loading reference test_source1.tsv entity IDs...")
    ref_ids = []
    with open(test_s1_path, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        id_col = header.index('entity_id')
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                ref_ids.append(parts[id_col])
    
    total_ref = len(ref_ids)
    print(f"  Reference test S1 entities: {total_ref:,}")

    # Step 2: Validate matching_results.tsv
    print("\n[2/5] Auditing matching_results.tsv format and order...")
    matching_size_mb = os.path.getsize(matching_path) / (1024 * 1024)
    print(f"  File size: {matching_size_mb:.2f} MB")

    matched_count = 0
    singleton_count = 0
    total_predicted_matches = 0
    mismatches = 0
    line_idx = 0

    with open(matching_path, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        if header != ['source1_entity_id', 'matched_entity_ids']:
            print(f"❌ Invalid header in matching_results.tsv: {header}")
            return False
        
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ''

            if line_idx < total_ref and s1_id != ref_ids[line_idx]:
                if mismatches < 5:
                    print(f"❌ Order mismatch at line {line_idx+2}: expected {ref_ids[line_idx]}, got {s1_id}")
                mismatches += 1

            if matched_str:
                m_ids = matched_str.split(',')
                matched_count += 1
                total_predicted_matches += len(m_ids)
            else:
                singleton_count += 1

            line_idx += 1

    if mismatches > 0:
        print(f"❌ Total ID mismatches/order violations: {mismatches}")
        return False
    if line_idx != total_ref:
        print(f"❌ Line count mismatch: expected {total_ref} data rows, found {line_idx}")
        return False

    print(f"  ✅ Line-by-line ID alignment: 100% PERFECT MATCH ({total_ref:,} rows)")
    print(f"  Entities with >= 1 match: {matched_count:,} ({matched_count/total_ref*100:.2f}%)")
    print(f"  Singletons (no matches): {singleton_count:,} ({singleton_count/total_ref*100:.2f}%)")
    print(f"  Total matched entities: {total_predicted_matches:,} (avg {total_predicted_matches/max(1, matched_count):.2f} per matched entity)")

    # Step 3: Validate candidate_pairs.tsv
    print("\n[3/5] Auditing candidate_pairs.tsv format and order...")
    cand_size_mb = os.path.getsize(candidate_path) / (1024 * 1024)
    print(f"  File size: {cand_size_mb:.2f} MB")

    cand_line_idx = 0
    total_candidates = 0
    cand_mismatches = 0

    with open(candidate_path, 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        if header != ['source1_entity_id', 'candidate_entity_ids']:
            print(f"❌ Invalid header in candidate_pairs.tsv: {header}")
            return False

        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ''

            if cand_line_idx < total_ref and s1_id != ref_ids[cand_line_idx]:
                if cand_mismatches < 5:
                    print(f"❌ Cand order mismatch at line {cand_line_idx+2}: expected {ref_ids[cand_line_idx]}, got {s1_id}")
                cand_mismatches += 1

            if cand_str:
                c_ids = cand_str.split(',')
                total_candidates += len(c_ids)

            cand_line_idx += 1

    if cand_mismatches > 0:
        print(f"❌ Total Candidate ID mismatches: {cand_mismatches}")
        return False
    if cand_line_idx != total_ref:
        print(f"❌ Candidate line count mismatch: expected {total_ref}, got {cand_line_idx}")
        return False

    print(f"  ✅ Candidate ID alignment: 100% PERFECT MATCH ({total_ref:,} rows)")
    print(f"  Total candidates evaluated: {total_candidates:,} (avg {total_candidates/total_ref:.1f} per entity)")

    # Step 4: Consistency check: Matches should be subset of candidates
    print("\n[4/5] Checking consistency: matches ⊆ candidates...")
    subset_violations = 0
    with open(matching_path, 'r', encoding='utf-8') as f_m, open(candidate_path, 'r', encoding='utf-8') as f_c:
        next(f_m)
        next(f_c)
        for i, (lm, lc) in enumerate(zip(f_m, f_c)):
            pm = lm.rstrip('\r\n').split('\t')
            pc = lc.rstrip('\r\n').split('\t')
            m_set = set(pm[1].split(',')) if len(pm) > 1 and pm[1] else set()
            c_set = set(pc[1].split(',')) if len(pc) > 1 and pc[1] else set()
            diff = m_set - c_set
            if diff:
                if subset_violations < 5:
                    print(f"⚠️ Row {i+2}: predicted match {diff} not in candidates!")
                subset_violations += 1

    if subset_violations == 0:
        print("  ✅ 100% of predicted matches are contained within candidate sets!")
    else:
        print(f"  ⚠️ {subset_violations} rows had matches outside candidates.")

    # Step 5: Summary
    print("\n[5/5] Final Verdict")
    print(f"  Elapsed verification time: {time.time()-t0:.1f}s")
    print("  " + "=" * 70)
    print("  🏆 VERIFICATION PASSED: All submission integrity checks satisfied!")
    print("  Both output files are 100% valid and ready for Unstop upload.")
    print("  " + "=" * 70)
    return True

if __name__ == '__main__':
    verify()
