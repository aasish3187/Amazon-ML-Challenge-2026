import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Read 50k rows from S1, S2, S3 with just entity_id and country
s1_c = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", nrows=50000, usecols=['entity_id', 'country'])
s1_map = dict(zip(s1_c['entity_id'], s1_c['country']))

s2_c = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t", nrows=200000, usecols=['entity_id', 'country'])
target_map = dict(zip(s2_c['entity_id'], s2_c['country']))

s3_c = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t", nrows=200000, usecols=['entity_id', 'country'])
target_map.update(dict(zip(s3_c['entity_id'], s3_c['country'])))

gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=50000)

cross_country_matches = 0
checked_pairs = 0
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    mids = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
    s1_country = s1_map.get(s1_id)
    if not s1_country:
        continue
    for mid in mids:
        mid = mid.strip()
        if not mid:
            continue
        m_country = target_map.get(mid)
        if m_country:
            checked_pairs += 1
            if m_country != s1_country:
                cross_country_matches += 1

print(f"Checked {checked_pairs:,} matched pairs.")
print(f"Cross-country matches: {cross_country_matches}")
