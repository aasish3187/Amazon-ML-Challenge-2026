import sys
sys.path.insert(0, '.')
from src.evaluate import compute_f05_per_entity

# Let's test the candidate pool on train
# Load 1000 train S1 entities, query 15 from S2 and 15 from S3 using candidate selection
# and check how many true matches are in the candidate set
print("Checking candidate recall on training data...")
