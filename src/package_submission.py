"""
Submission Packaging Script
Zips output/matching_results.tsv and candidate_pairs.tsv into a clean submission zip
ready for leaderboard upload.
"""
import os
import sys
import zipfile
import time

sys.stdout.reconfigure(encoding='utf-8')

def package_submission():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output")
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    zip_path = os.path.join(output_dir, "submission.zip")

    if not os.path.exists(matching_file):
        print(f"❌ Error: {matching_file} does not exist yet.")
        return False

    print(f"📦 Packaging submission into {zip_path}...")
    t0 = time.time()
    
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        print("  Adding matching_results.tsv...")
        zipf.write(matching_file, arcname="matching_results.tsv")
        if os.path.exists(candidate_file):
            print("  Adding candidate_pairs.tsv...")
            zipf.write(candidate_file, arcname="candidate_pairs.tsv")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"✅ Created {zip_path} ({size_mb:.2f} MB) in {time.time()-t0:.1f}s")
    return True

if __name__ == '__main__':
    package_submission()
