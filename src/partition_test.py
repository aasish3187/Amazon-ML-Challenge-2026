"""
Stream and partition test data by country into disk files.
Zero RAM consumption, fast streaming in one pass.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

sys.stdout.reconfigure(encoding='utf-8')


def partition_test_dataset(output_base: str = "partitions/test"):
    print("=" * 70)
    print("  PARTITIONING TEST DATASET BY COUNTRY (STREAMING)")
    print("=" * 70)
    t0 = time.time()

    os.makedirs(output_base, exist_ok=True)

    # 1. Partition S1
    print(f"\n[1/3] Partitioning test_source1.tsv...")
    s1_files = {}
    s1_counts = {}

    with open(TEST_S1, 'r', encoding='utf-8') as f:
        header = next(f)
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            country = parts[3].strip()
            if country not in s1_files:
                cdir = os.path.join(output_base, country)
                os.makedirs(cdir, exist_ok=True)
                s1_files[country] = open(os.path.join(cdir, "s1.tsv"), 'w', encoding='utf-8')
                s1_files[country].write(header)
                s1_counts[country] = 0

            s1_files[country].write(line)
            s1_counts[country] += 1
            if (i + 1) % 500000 == 0:
                print(f"    Processed {i+1:,} S1 rows...")

    for f in s1_files.values():
        f.close()

    print("  S1 Breakdown:")
    for c, cnt in s1_counts.items():
        print(f"    {c:10s}: {cnt:,} entities")

    # 2. Partition Pool (S2 + S3)
    print(f"\n[2/3] Partitioning test_source2.tsv & test_source3.tsv into per-country pools...")
    pool_files = {}
    pool_counts = {}

    for path in [TEST_S2, TEST_S3]:
        fname = os.path.basename(path)
        print(f"  Streaming {fname}...")
        with open(path, 'r', encoding='utf-8') as f:
            header = next(f)
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                country = parts[3].strip()
                if country not in pool_files:
                    cdir = os.path.join(output_base, country)
                    os.makedirs(cdir, exist_ok=True)
                    pool_files[country] = open(os.path.join(cdir, "pool.tsv"), 'w', encoding='utf-8')
                    pool_files[country].write(header)
                    pool_counts[country] = 0

                pool_files[country].write(line)
                pool_counts[country] += 1
                if (i + 1) % 1000000 == 0:
                    print(f"    [{fname}] Processed {i+1:,} rows...")

    for f in pool_files.values():
        f.close()

    print("\n  Pool Breakdown:")
    for c, cnt in pool_counts.items():
        print(f"    {c:10s}: {cnt:,} candidate records")

    print(f"\n{'='*70}")
    print(f"  ✅ Test partitioning complete in {(time.time()-t0)/60:.1f} minutes!")
    print(f"{'='*70}")


if __name__ == '__main__':
    partition_test_dataset()
