"""
DEEP DIAGNOSIS: Understand exactly WHY we're at 0.467 and what's needed for 0.99+
"""
import os, sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from src.config import *
from src.evaluate import compute_f05_per_entity, parse_ground_truth, parse_predictions
from collections import Counter

# 1. Load ground truth
print("="*70)
print("  DEEP DIAGNOSIS — WHERE IS THE SCORE LEAKING?")
print("="*70)

gt = parse_ground_truth(TRAIN_GT)
print(f"\nGround truth: {len(gt):,} S1 entities")

true_singletons = {k for k, v in gt.items() if len(v) == 0}
true_matched = {k for k, v in gt.items() if len(v) > 0}
print(f"  True singletons: {len(true_singletons):,} ({len(true_singletons)/len(gt)*100:.1f}%)")
print(f"  True matched:    {len(true_matched):,} ({len(true_matched)/len(gt)*100:.1f}%)")

# Distribution of match counts
match_counts = Counter(len(v) for v in gt.values())
print(f"\n  Match count distribution:")
for cnt, freq in sorted(match_counts.items()):
    print(f"    {cnt} matches: {freq:,} entities ({freq/len(gt)*100:.2f}%)")

# 2. Country breakdown - read train_source1 to find country
print(f"\n--- Country analysis from training data ---")
s1_country = {}
with open(TRAIN_S1, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 3:
            s1_country[parts[0]] = parts[3].strip()

# Check unique countries
unique_countries = set(s1_country.values())
print(f"  Unique countries in train S1: {unique_countries}")

for c in sorted(unique_countries):
    c_ids = {sid for sid, co in s1_country.items() if co == c and sid in gt}
    if not c_ids:
        print(f"\n  {c}: No overlap with ground truth")
        continue
    c_singletons = {sid for sid in c_ids if len(gt[sid]) == 0}
    c_matched = {sid for sid in c_ids if len(gt[sid]) > 0}
    c_match_counts = [len(gt[sid]) for sid in c_matched]
    avg_matches = sum(c_match_counts) / len(c_match_counts) if c_match_counts else 0
    print(f"\n  {c}:")
    print(f"    Total S1: {len(c_ids):,}")
    print(f"    Singletons: {len(c_singletons):,} ({len(c_singletons)/len(c_ids)*100:.1f}%)")
    print(f"    Matched: {len(c_matched):,} ({len(c_matched)/len(c_ids)*100:.1f}%)")
    print(f"    Avg matches per matched entity: {avg_matches:.2f}")

# 3. True Match Source Analysis
print(f"\n--- True Match Source Analysis ---")
all_true_matches = set()
for v in gt.values():
    all_true_matches.update(v)

s2_true = {m for m in all_true_matches if m.startswith('S2-')}
s3_true = {m for m in all_true_matches if m.startswith('S3-')}
print(f"  Total unique true match IDs: {len(all_true_matches):,}")
print(f"    From S2: {len(s2_true):,}")
print(f"    From S3: {len(s3_true):,}")

# 4. What's the typical match pattern? (S2 only, S3 only, both?)
both_sources = 0
s2_only = 0
s3_only = 0
for s1_id, matches in gt.items():
    if not matches:
        continue
    has_s2 = any(m.startswith('S2-') for m in matches)
    has_s3 = any(m.startswith('S3-') for m in matches)
    if has_s2 and has_s3:
        both_sources += 1
    elif has_s2:
        s2_only += 1
    elif has_s3:
        s3_only += 1
print(f"\n  Match pattern (among {len(true_matched):,} matched entities):")
print(f"    Both S2+S3: {both_sources:,} ({both_sources/len(true_matched)*100:.1f}%)")
print(f"    S2 only:    {s2_only:,} ({s2_only/len(true_matched)*100:.1f}%)")
print(f"    S3 only:    {s3_only:,} ({s3_only/len(true_matched)*100:.1f}%)")

# 5. Submission Output Analysis
print(f"\n{'='*70}")
print(f"  SUBMISSION OUTPUT ANALYSIS")
print(f"{'='*70}")

pred = parse_predictions(MATCHING_OUTPUT)
print(f"  Total predictions: {len(pred):,}")
pred_empty = sum(1 for v in pred.values() if len(v) == 0)
pred_matched = sum(1 for v in pred.values() if len(v) > 0)
print(f"  Predicted singletons: {pred_empty:,} ({pred_empty/len(pred)*100:.1f}%)")
print(f"  Predicted matched: {pred_matched:,} ({pred_matched/len(pred)*100:.1f}%)")

# Avg predictions per matched entity
match_counts_pred = [len(v) for v in pred.values() if len(v) > 0]
avg_m = sum(match_counts_pred)/len(match_counts_pred) if match_counts_pred else 0
print(f"  Avg matches per matched entity: {avg_m:.2f}")

pred_match_dist = Counter(len(v) for v in pred.values())
print(f"\n  Prediction match count distribution:")
for cnt, freq in sorted(pred_match_dist.items()):
    pct = freq/len(pred)*100
    print(f"    {cnt} matches: {freq:,} ({pct:.2f}%)")

# 6. Per-country predictions
print(f"\n--- Per-Country Prediction Analysis ---")
for c in ['France', 'US', 'India']:
    c_path = os.path.join(OUTPUT_DIR, f"matching_{c}.tsv")
    if os.path.exists(c_path):
        c_pred = parse_predictions(c_path)
        c_empty = sum(1 for v in c_pred.values() if len(v) == 0)
        c_match = sum(1 for v in c_pred.values() if len(v) > 0)
        match_counts_c = [len(v) for v in c_pred.values() if len(v) > 0]
        avg_m_c = sum(match_counts_c)/len(match_counts_c) if match_counts_c else 0
        print(f"\n  {c}:")
        print(f"    Total: {len(c_pred):,}")
        print(f"    Empty/Singleton: {c_empty:,} ({c_empty/len(c_pred)*100:.1f}%)")
        print(f"    Matched: {c_match:,} ({c_match/len(c_pred)*100:.1f}%)")
        print(f"    Avg matches per matched entity: {avg_m_c:.2f}")

# 7. CRITICAL INSIGHT: The gap between GT and our predictions
print(f"\n{'='*70}")
print(f"  CRITICAL GAP ANALYSIS")
print(f"{'='*70}")

# GT says 94.4% matched, we predicted only X% matched
# GT says avg 3.46 matches, we predicted avg Y
gt_avg_matches = sum(len(v) for v in gt.values() if len(v) > 0) / len(true_matched)
print(f"\n  Ground truth: 94.4% matched, avg {gt_avg_matches:.2f} matches per entity")
print(f"  Our prediction: {pred_matched/len(pred)*100:.1f}% matched, avg {avg_m:.2f} matches per entity")
print(f"\n  GAP in singleton rate: {pred_empty/len(pred)*100 - 5.6:.1f}% too many singletons")
print(f"  GAP in avg matches: {gt_avg_matches - avg_m:.2f} missing matches per entity")

# 8. Estimate score impact
print(f"\n--- Score Impact Estimation ---")
# Each false singleton (true matched but we predict empty) scores 0.0
# With 94.4% truly matched, predicting X% singletons means (X-5.6)% are false singletons
false_singleton_rate = max(0, pred_empty/len(pred) - 0.056)
print(f"  Estimated false singleton rate: {false_singleton_rate*100:.1f}%")
print(f"  These score 0.0 each, pulling down the macro average")
print(f"  Even if matched entities score perfectly (1.0), overall would be:")
print(f"    {1.0 - false_singleton_rate:.4f}")

# 9. Our blocking is capped at 20+20=40 candidates per entity.
# GT shows avg 3.46 matches. With 40 candidates, blocking recall needs to be high.
print(f"\n--- Blocking Capacity Analysis ---")
print(f"  Current: 20 candidates from S2 + 20 from S3 = 40 max")
print(f"  GT avg matches: {gt_avg_matches:.2f}")
print(f"  If blocking recall is ~80%, we find {gt_avg_matches*0.8:.2f} of {gt_avg_matches:.2f} true matches")
print(f"  This caps entity F0.5 at ~0.80 even with perfect classification")

print(f"\n{'='*70}")
print(f"  DIAGNOSIS COMPLETE")
print(f"{'='*70}")
