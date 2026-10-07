import sys
sys.stdout.reconfigure(encoding='utf-8')

targets = {
    'S1-363633018': ['S2-467145647', 'S3-121575506'],
    'S1-261899496': ['S3-757141290', 'S2-478982955', 'S2-274873975'],
    'S1-618318002': ['S3-462769526', 'S3-28959576']
}
all_ids = set(targets.keys())
for v in targets.values():
    all_ids.update(v)

records = {}
for path in ['student_resource/dataset/train/train_source1.tsv', 'student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split('\t')
            if p[0] in all_ids:
                records[p[0]] = p

for s1_id, matches in targets.items():
    s1 = records.get(s1_id)
    print("=" * 60)
    print(f"S1: {s1_id} | name: {s1[1]} | addr: {s1[2] if len(s1)>2 else ''}")
    for m in matches:
        rec = records.get(m)
        if rec:
            print(f"  Match: {m} | name: {rec[1]} | addr: {rec[2] if len(rec)>2 else ''}")
        else:
            print(f"  Match: {m} NOT FOUND")
