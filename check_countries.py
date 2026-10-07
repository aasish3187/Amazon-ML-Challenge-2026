import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("--- Country distributions in Train (first 200k rows each) ---")
for name in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    df = pd.read_csv(f"student_resource/dataset/train/{name}", sep="\t", nrows=200000)
    print(f"\n{name} countries:")
    print(df['country'].value_counts(dropna=False).to_string())

print("\n--- Country distributions in Test (first 200k rows each) ---")
for name in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
    df = pd.read_csv(f"student_resource/dataset/test/{name}", sep="\t", nrows=200000)
    print(f"\n{name} countries:")
    print(df['country'].value_counts(dropna=False).to_string())
