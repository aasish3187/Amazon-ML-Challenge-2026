"""
Deep analysis of training ground truth to understand match patterns,
then evaluate our current approach against training data.
"""
import sys
import time
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

def main():
    t0 = time.time()
    print("=" * 70)
    print("  DEEP GROUND TRUTH ANALYSIS")
    print("=" * 70)

    # 1. Analyze ground truth distribution
    print("\n[1] Loading ground truth...")
    gt = {}
    with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            sid = parts[0]
            if len(parts) > 1 and parts[1].strip():
                matches = [x.strip() for x in parts[1].split(',') if x.strip()]
            else:
                matches = []
            gt[sid] = matches

    total = len(gt)
    singletons = sum(1 for m in gt.values() if len(m) == 0)
    has_matches = sum(1 for m in gt.values() if len(m) > 0)

    print(f"  Total S1 entities: {total:,}")
    print(f"  Singletons (no matches): {singletons:,} ({singletons/total*100:.2f}%)")
    print(f"  Entities with matches: {has_matches:,} ({has_matches/total*100:.2f}%)")

    # Match count distribution
    match_counts = Counter(len(m) for m in gt.values())
    print(f"\n  Match count distribution:")
    for k in sorted(match_counts.keys()):
        pct = match_counts[k] / total * 100
        print(f"    {k} matches: {match_counts[k]:,} ({pct:.2f}%)")

    # Source distribution in matches
    s2_only = 0
    s3_only = 0
    both = 0
    s2_count = 0
    s3_count = 0
    for m in gt.values():
        if not m:
            continue
        has_s2 = any(x.startswith('S2-') for x in m)
        has_s3 = any(x.startswith('S3-') for x in m)
        s2_count += sum(1 for x in m if x.startswith('S2-'))
        s3_count += sum(1 for x in m if x.startswith('S3-'))
        if has_s2 and has_s3:
            both += 1
        elif has_s2:
            s2_only += 1
        elif has_s3:
            s3_only += 1

    print(f"\n  Source composition of matched entities:")
    print(f"    S2-only matches: {s2_only:,} ({s2_only/has_matches*100:.2f}%)")
    print(f"    S3-only matches: {s3_only:,} ({s3_only/has_matches*100:.2f}%)")
    print(f"    Both S2+S3: {both:,} ({both/has_matches*100:.2f}%)")
    print(f"    Total S2 IDs in matches: {s2_count:,}")
    print(f"    Total S3 IDs in matches: {s3_count:,}")

    # 2. Load source data to understand name/address patterns
    print(f"\n[2] Loading training source data (samples)...")

    # Load S1
    s1_data = {}
    with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        header = next(f).strip().split('\t')
        print(f"  S1 columns: {header}")
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                s1_data[parts[0]] = {'name': parts[1], 'address': parts[2], 'country': parts[3] if len(parts) > 3 else ''}

    # Load S2
    s2_data = {}
    with open('student_resource/dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                s2_data[parts[0]] = {'name': parts[1], 'address': parts[2], 'country': parts[3] if len(parts) > 3 else ''}

    # Load S3
    s3_data = {}
    with open('student_resource/dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                s3_data[parts[0]] = {'name': parts[1], 'address': parts[2], 'country': parts[3] if len(parts) > 3 else ''}

    print(f"  S1 records: {len(s1_data):,}")
    print(f"  S2 records: {len(s2_data):,}")
    print(f"  S3 records: {len(s3_data):,}")

    # Country distribution
    s1_countries = Counter(d['country'] for d in s1_data.values())
    print(f"\n  S1 country distribution: {dict(s1_countries)}")

    # 3. Analyze match quality patterns - what do true matches look like?
    print(f"\n[3] Analyzing TRUE MATCH similarity patterns (first 50k matched entities)...")

    from rapidfuzz import fuzz
    import re

    def extract_numbers(text):
        return set(re.findall(r'\b\d+\b', text))

    pool = {**s2_data, **s3_data}

    name_ratios = []
    addr_ratios = []
    number_overlap_rates = []
    country_match_rates = []
    sample_count = 0
    examples = []

    for sid, matches in gt.items():
        if not matches:
            continue
        if sid not in s1_data:
            continue
        s1 = s1_data[sid]

        for mid in matches:
            if mid not in pool:
                continue
            m = pool[mid]

            nr = fuzz.token_sort_ratio(s1['name'].lower(), m['name'].lower())
            ar = fuzz.token_sort_ratio(s1['address'].lower(), m['address'].lower())
            name_ratios.append(nr)
            addr_ratios.append(ar)

            s1_nums = extract_numbers(s1['address'])
            m_nums = extract_numbers(m['address'])
            if s1_nums and m_nums:
                overlap = len(s1_nums & m_nums) / max(len(s1_nums), len(m_nums))
                number_overlap_rates.append(overlap)

            country_match_rates.append(1 if s1['country'] == m['country'] else 0)

            if sample_count < 20:
                examples.append({
                    's1_name': s1['name'][:60],
                    's1_addr': s1['address'][:60],
                    'm_name': m['name'][:60],
                    'm_addr': m['address'][:60],
                    'name_ratio': nr,
                    'addr_ratio': ar,
                    'mid': mid
                })

            sample_count += 1
            if sample_count >= 100000:
                break
        if sample_count >= 100000:
            break

    print(f"  Analyzed {sample_count:,} true match pairs")
    print(f"\n  NAME similarity (token_sort_ratio):")
    print(f"    Mean: {sum(name_ratios)/len(name_ratios):.1f}")
    print(f"    Median: {sorted(name_ratios)[len(name_ratios)//2]}")
    print(f"    <50: {sum(1 for x in name_ratios if x < 50)/len(name_ratios)*100:.1f}%")
    print(f"    <60: {sum(1 for x in name_ratios if x < 60)/len(name_ratios)*100:.1f}%")
    print(f"    <70: {sum(1 for x in name_ratios if x < 70)/len(name_ratios)*100:.1f}%")
    print(f"    <80: {sum(1 for x in name_ratios if x < 80)/len(name_ratios)*100:.1f}%")
    print(f"    >=80: {sum(1 for x in name_ratios if x >= 80)/len(name_ratios)*100:.1f}%")
    print(f"    >=90: {sum(1 for x in name_ratios if x >= 90)/len(name_ratios)*100:.1f}%")

    print(f"\n  ADDRESS similarity (token_sort_ratio):")
    print(f"    Mean: {sum(addr_ratios)/len(addr_ratios):.1f}")
    print(f"    Median: {sorted(addr_ratios)[len(addr_ratios)//2]}")
    print(f"    <50: {sum(1 for x in addr_ratios if x < 50)/len(addr_ratios)*100:.1f}%")
    print(f"    <70: {sum(1 for x in addr_ratios if x < 70)/len(addr_ratios)*100:.1f}%")
    print(f"    >=80: {sum(1 for x in addr_ratios if x >= 80)/len(addr_ratios)*100:.1f}%")

    if number_overlap_rates:
        print(f"\n  ADDRESS number overlap rate (when both have numbers):")
        print(f"    Mean: {sum(number_overlap_rates)/len(number_overlap_rates):.3f}")
        print(f"    Zero overlap: {sum(1 for x in number_overlap_rates if x == 0)/len(number_overlap_rates)*100:.1f}%")
        print(f"    Full overlap: {sum(1 for x in number_overlap_rates if x >= 1.0)/len(number_overlap_rates)*100:.1f}%")

    print(f"\n  Country match rate: {sum(country_match_rates)/len(country_match_rates)*100:.1f}%")

    # 4. Sample true matches to see patterns
    print(f"\n[4] Sample TRUE MATCHES:")
    for i, ex in enumerate(examples[:15]):
        print(f"\n  Example {i+1}: ({ex['mid']})")
        print(f"    S1 name: {ex['s1_name']}")
        print(f"    S1 addr: {ex['s1_addr']}")
        print(f"    M  name: {ex['m_name']}")
        print(f"    M  addr: {ex['m_addr']}")
        print(f"    Name ratio: {ex['name_ratio']} | Addr ratio: {ex['addr_ratio']}")

    # 5. Now let's check what our CURRENT submission looks like vs ground truth
    # We can evaluate our approach on training data
    print(f"\n[5] Evaluating our scoring approach against TRAINING ground truth...")
    print(f"    (This tells us our theoretical maximum)")

    # Check how many S1 training entities have matches in pool
    gt_matched = {sid: m for sid, m in gt.items() if m}
    all_match_ids = set()
    for m in gt_matched.values():
        all_match_ids.update(m)

    s2_ids_in_gt = sum(1 for x in all_match_ids if x.startswith('S2-'))
    s3_ids_in_gt = sum(1 for x in all_match_ids if x.startswith('S3-'))
    print(f"    Unique S2 IDs in ground truth: {s2_ids_in_gt:,}")
    print(f"    Unique S3 IDs in ground truth: {s3_ids_in_gt:,}")
    print(f"    S2 records available: {len(s2_data):,}")
    print(f"    S3 records available: {len(s3_data):,}")

    # Check if any GT IDs are missing from source files
    missing = sum(1 for x in all_match_ids if x not in pool)
    print(f"    GT IDs missing from source files: {missing:,}")

    # 6. Quick simulation: what if we get PERFECT precision but varying recall?
    print(f"\n[6] Score simulation (Macro F0.5):")
    for recall_pct in [50, 60, 70, 80, 90, 95, 99, 100]:
        # Perfect precision, partial recall
        scores = []
        for sid, matches in gt.items():
            if not matches:
                # True singleton, we predict singleton correctly
                scores.append(1.0)
            else:
                # True matches exist
                n_true = len(matches)
                n_predicted = max(1, int(n_true * recall_pct / 100))
                # Perfect precision = all predicted are correct
                precision = 1.0
                recall = n_predicted / n_true
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                scores.append(f05)
        macro = sum(scores) / len(scores)
        print(f"    Recall={recall_pct}% + Perfect Precision -> Macro F0.5 = {macro:.6f}")

    # What if we predict NO matches at all (all singletons)?
    scores_all_single = []
    for sid, matches in gt.items():
        if not matches:
            scores_all_single.append(1.0)
        else:
            scores_all_single.append(0.0)
    print(f"\n    ALL SINGLETONS (predict nothing) -> Macro F0.5 = {sum(scores_all_single)/len(scores_all_single):.6f}")

    elapsed = time.time() - t0
    print(f"\n  Analysis complete in {elapsed:.1f}s")

if __name__ == '__main__':
    main()
