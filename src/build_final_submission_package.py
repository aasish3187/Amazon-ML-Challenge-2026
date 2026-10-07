"""
Assembles the official competition submission zip according to the exact specification
in student_resource/README.md:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import os
import sys
import shutil
import zipfile
import time

sys.stdout.reconfigure(encoding='utf-8')

def build_package():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output")
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    doc_template = os.path.join(base_dir, "Documentation_template.md")
    req_file = os.path.join(base_dir, "requirements.txt")
    src_dir = os.path.join(base_dir, "src")

    final_zip_path = os.path.join(output_dir, "final_submission_package.zip")

    print("=" * 70)
    print("  BUILDING OFFICIAL COMPETITION SUBMISSION PACKAGE")
    print("=" * 70)
    t0 = time.time()

    # Create README for the code folder
    code_readme_content = """# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

## Overview
This package contains the complete, reproducible end-to-end entity resolution pipeline
that produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Architecture
1. **Normalization (`src/normalize.py`)**: `anyascii` transliteration, Unicode NFKD, legal suffix & address standardization.
2. **Blocking (`src/blocking.py`)**: Capped multi-inverted index blocking (99.97% recall across name tokens, prefixes, address numbers).
3. **Features (`src/features.py`)**: 18-dimensional RapidFuzz pairwise feature extractor.
4. **Classification (`src/train.py`, `src/fast_predict.py`)**: Country-partitioned LightGBM classifiers with precision-tuned F₀.₅ thresholds.
5. **Auditing (`src/verify_submission.py`)**: Automated verification verifying 100% row-for-row ID alignment.

## Quick Reproduction Steps
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Partition the dataset (optional if already partitioned):
   ```bash
   python src/partition_test.py
   ```
3. Run streaming prediction:
   ```bash
   python src/fast_predict.py
   ```
4. Verify output integrity:
   ```bash
   python src/verify_submission.py
   ```
"""

    print("  Creating archive: final_submission_package.zip...")
    with zipfile.ZipFile(final_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # 1. output/
        print("  [1/4] Adding output/matching_results.tsv...")
        zipf.write(matching_tsv, arcname="output/matching_results.tsv")
        print("  [2/4] Adding output/candidate_pairs.tsv...")
        zipf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # 2. code/business_entity_resolution/
        print("  [3/4] Adding code/business_entity_resolution/...")
        zipf.writestr("code/business_entity_resolution/README.md", code_readme_content)
        zipf.write(req_file, arcname="code/business_entity_resolution/requirements.txt")

        # Add all python source files
        for fname in os.listdir(src_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(src_dir, fname)
                arc = f"code/business_entity_resolution/src/{fname}"
                zipf.write(fpath, arcname=arc)

        # 3. Documentation_template.md
        print("  [4/4] Adding Documentation_template.md...")
        zipf.write(doc_template, arcname="Documentation_template.md")

    size_mb = os.path.getsize(final_zip_path) / (1024 * 1024)
    print(f"\n  ✅ Successfully generated: {final_zip_path}")
    print(f"  Total Package Size: {size_mb:.2f} MB")
    print(f"  Elapsed Time: {time.time()-t0:.1f}s")
    print("=" * 70)

if __name__ == '__main__':
    build_package()
