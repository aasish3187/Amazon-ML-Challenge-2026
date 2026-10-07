import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Read first 1000 rows of S1
s1_df = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", nrows=1000)
s1_id_set = set(s1_df['entity_id'])

# Stream ground truth to find matches for these S1 IDs
matched_rows = []
with open("student_resource/dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[0] in s1_id_set and parts[1]:
            matched_rows.append((parts[0], parts[1]))
            if len(matched_rows) >= 5:
                break

all_target_ids = set()
for s1_id, m_str in matched_rows:
    for x in m_str.split(','):
        all_target_ids.add(x.strip())

s1_dict = s1_df[s1_df['entity_id'].isin({x[0] for x in matched_rows})].set_index('entity_id').to_dict('index')

# Stream S2 and S3 to find those target IDs
target_dict = {}
for src, filename in [("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")]:
    with open(f"student_resource/dataset/train/{filename}", "r", encoding="utf-8") as f:
        header = next(f).strip().split("\t")
        id_idx = header.index("entity_id")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(id_idx, name_idx, addr_idx):
                eid = parts[id_idx]
                if eid in all_target_ids:
                    target_dict[eid] = {
                        "business_name": parts[name_idx],
                        "business_address": parts[addr_idx]
                    }
                    if len(target_dict) >= len(all_target_ids):
                        break

print("="*80)
print("REAL MATCH EXAMPLES FROM GROUND TRUTH")
print("="*80)
for s1_id, m_str in matched_rows:
    mids = [x.strip() for x in m_str.split(',') if x.strip()]
    s1_info = s1_dict[s1_id]
    print(f"\n[SOURCE 1] ID: {s1_id} | Country: {s1_info.get('country')}")
    print(f"  Name   : {s1_info.get('business_name')}")
    print(f"  Address: {s1_info.get('business_address')}")
    print("  MATCHES:")
    for mid in mids:
        if mid in target_dict:
            t_info = target_dict[mid]
            print(f"    -> [{mid[:2]}] {mid}")
            print(f"       Name   : {t_info.get('business_name')}")
            print(f"       Address: {t_info.get('business_address')}")
        else:
            print(f"    -> [{mid[:2]}] {mid} (still searching)")
