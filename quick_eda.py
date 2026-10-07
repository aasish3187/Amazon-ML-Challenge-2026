import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

base_dir = "student_resource/dataset/train"

print("--- Sample train_source1.tsv ---")
s1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep="\t", nrows=5)
print(s1.to_string())

print("\n--- Sample train_source2.tsv ---")
s2 = pd.read_csv(os.path.join(base_dir, "train_source2.tsv"), sep="\t", nrows=5)
print(s2.to_string())

print("\n--- Sample train_source3.tsv ---")
s3 = pd.read_csv(os.path.join(base_dir, "train_source3.tsv"), sep="\t", nrows=5)
print(s3.to_string())

print("\n--- Sample train_ground_truth.tsv ---")
gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep="\t", nrows=10)
print(gt.to_string())
