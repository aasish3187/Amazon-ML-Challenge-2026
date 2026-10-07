import sys
sys.path.insert(0, '.')
from src.evaluate import parse_ground_truth

gt = parse_ground_truth('student_resource/dataset/train/train_ground_truth.tsv')

s1_recs = {}
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        if len(s1_recs) < 15:
            s1_recs[p[0]] = (p[1], p[2] if len(p)>2 else '', p[3])
        else:
            break

needed = set()
for sid in s1_recs:
    needed.update(gt.get(sid, set()))

pool_recs = {}
for p in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    with open(p, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in needed:
                pool_recs[parts[0]] = (parts[1], parts[2] if len(parts)>2 else '')

for sid, (s1_name, s1_addr, country) in s1_recs.items():
    matches = gt.get(sid, set())
    print(f"=== S1 [{sid}] ({country}) ===")
    print(f"  Name: {s1_name}")
    print(f"  Addr: {s1_addr}")
    print(f"  True Matches ({len(matches)}):")
    for m in matches:
        if m in pool_recs:
            m_name, m_addr = pool_recs[m]
            print(f"    [{m}] Name: \"{m_name}\" | Addr: \"{m_addr}\"")
    print()
