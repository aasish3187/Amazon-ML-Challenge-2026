import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

train_dir = "student_resource/dataset/train"
test_dir = "student_resource/dataset/test"

def count_lines(path):
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            count += 1
    return count - 1  # exclude header

print("Counting dataset rows...")
for d, name in [(train_dir, "TRAIN"), (test_dir, "TEST")]:
    print(f"\n--- {name} SET ---")
    for f in sorted(os.listdir(d)):
        if f.endswith(".tsv"):
            p = os.path.join(d, f)
            print(f"{f}: {count_lines(p):,} rows ({os.path.getsize(p)/1e6:.1f} MB)")
