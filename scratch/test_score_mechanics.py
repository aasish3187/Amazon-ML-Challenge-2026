import sys, os, pickle, random
import numpy as np

sys.path.insert(0, '.')
from src.evaluate import compute_f05_per_entity

# Load training ground truth
gt = {}
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        gt[p[0]] = set(p[1].split(',')) if len(p) > 1 and p[1].strip() else set()

# In task-3630.log, we ran ml_diagnostic.py on 3000 training entities:
# 1188 India, 1812 US.
# Let's check why task-3630.log had Macro F0.5 = 0.580094!
# In task-3630.log:
# Perfect (>=0.99): 356/3000
# Zero: 702/3000 (23.4%)
# Notice that 23.4% of entities got 0.0!
# And what about the remaining 2,298 entities?
# If 702 got 0.0, and overall average is 0.580, then the average on the non-zero entities was:
# (0.580 * 3000) / (3000 - 702) = 1740 / 2298 = 0.757!
print(f"Non-zero average: {1740 / 2298:.4f}")
