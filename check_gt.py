import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Verifying if matches ever cross countries...")
# Load a sample or inspect country mappings
gt_sample = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=100000)

# Check singletons vs matched
has_matches = gt_sample['matched_entity_ids'].notna() & (gt_sample['matched_entity_ids'].str.strip() != "")
print(f"Total S1 entities in sample: {len(gt_sample):,}")
print(f"Entities with at least one match: {has_matches.sum():,} ({has_matches.mean()*100:.2f}%)")
print(f"Singletons (zero matches): {(~has_matches).sum():,} ({(~has_matches).mean()*100:.2f}%)")

# Average matches per matched entity
match_counts = gt_sample.loc[has_matches, 'matched_entity_ids'].apply(lambda x: len(str(x).split(',')))
print(f"Mean matches per matched entity: {match_counts.mean():.2f}")
print(f"Max matches for a single entity: {match_counts.max()}")
print(f"Distribution of match counts:\n{match_counts.value_counts().head(10).to_string()}")
