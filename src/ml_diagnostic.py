"""
DIAGNOSTIC: Test the actual ML pipeline against training ground truth
to find exactly where we're losing score.

Split: Sample 5k training S1 entities, run through our EXACT pipeline
(blocking + features + ML model), compare predictions vs ground truth.
"""
import os, sys, time, pickle, re, random
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from src.config import *
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.train_ensemble import EnsemblePredictor
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

STOPWORDS = {
    'the', 'and', 'company', 'corporation', 'limited', 'private', 'ltd',
    'pvt', 'llc', 'inc', 'services', 'management', 'group', 'solutions',
    'enterprises', 'international', 'holdings', 'global', 'technologies',
    'sarl', 'sasu', 'eurl', 'association'
}

COMMON_ADDR_STOP = {
    'the', 'and', 'road', 'rd', 'street', 'st', 'lane', 'ln', 'floor', 'fl',
    'building', 'bldg', 'near', 'opp', 'opposite', 'behind', 'beside', 'plot',
    'no', 'hno', 'dno', 'door', 'shop', 'flat', 'apartment', 'nagar', 'colony',
    'india', 'delhi', 'mumbai', 'bengaluru', 'hyderabad', 'chennai', 'kolkata',
    'state', 'city', 'west', 'east', 'north', 'south', 'cross', 'main', 'sector'
}

def build_index_uncapped(pool_records, country):
    """Build inverted index with NO cap (or very high cap)."""
    name_idx = defaultdict(list)
    num_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    
    MAX_CAP = 50000  # Very high cap to prevent memory explosion but not starve
    
    for i, rec in enumerate(pool_records):
        for tok in rec['name_tokens']:
            if len(tok) >= 3 and tok not in STOPWORDS:
                if len(name_idx[tok]) < MAX_CAP:
                    name_idx[tok].append(i)
        for num in rec['addr_numbers']:
            if len(num) >= 2:
                if len(num_idx[num]) < MAX_CAP:
                    num_idx[num].append(i)
        for tok in rec['norm_addr'].split():
            if len(tok) >= 5 and tok not in COMMON_ADDR_STOP:
                if len(addr_idx[tok]) < MAX_CAP:
                    addr_idx[tok].append(i)
    
    return name_idx, num_idx, addr_idx


def query_candidates(s1_rec, name_idx, num_idx, addr_idx, top_k=50):
    """Get top-k candidates."""
    sc = defaultdict(int)
    
    for tok in s1_rec['name_tokens']:
        if len(tok) >= 3 and tok not in STOPWORDS:
            w = 8 if len(tok) >= 5 else 5
            postings = name_idx.get(tok, ())
            if len(postings) > 3000:
                postings = postings[:3000]
            for pi in postings:
                sc[pi] += w
    
    for num in s1_rec['addr_numbers']:
        if len(num) >= 2:
            w = 12 if len(num) == 6 else 6
            postings = num_idx.get(num, ())
            if len(postings) > 3000:
                postings = postings[:3000]
            for pi in postings:
                sc[pi] += w
    
    for tok in s1_rec['norm_addr'].split():
        if len(tok) >= 5 and tok not in COMMON_ADDR_STOP:
            postings = addr_idx.get(tok, ())
            if len(postings) < 500:
                for pi in postings:
                    sc[pi] += 5
    
    if not sc:
        return []
    return [pi for pi, _ in sorted(sc.items(), key=lambda x: -x[1])[:top_k]]


def main():
    t0 = time.time()
    print("=" * 70)
    print("  ML PIPELINE DIAGNOSTIC ON TRAINING DATA")
    print("=" * 70)
    
    # Load models
    with open('output/models_ensemble.pkl', 'rb') as f:
        saved = pickle.load(f)
    models = saved['models']
    print(f"Models: {list(models.keys())}")
    
    # Load GT
    gt = {}
    with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            gt[p[0]] = set(x.strip() for x in p[1].split(',') if x.strip()) if len(p) > 1 and p[1].strip() else set()
    
    # Load S1 sample
    random.seed(42)
    all_s1 = []
    with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                all_s1.append(p)
    
    SAMPLE = 3000
    sample = random.sample(all_s1, min(SAMPLE, len(all_s1)))
    
    # Group by country
    by_country = defaultdict(list)
    for p in sample:
        by_country[p[3] if len(p) > 3 else 'US'].append(p)
    
    print(f"Sample: {len(sample):,} entities")
    for c, lst in by_country.items():
        print(f"  {c}: {len(lst):,}")
    
    # Process per country (like the real pipeline)
    all_scores = []
    all_debug = []
    
    for country, s1_list in by_country.items():
        print(f"\n--- {country} ({len(s1_list):,} entities) ---")
        tc = time.time()
        
        model = models.get(country, models['US'])
        
        # Load pool for this country
        pool = []
        pool_id_to_idx = {}
        for src in ['train_source2.tsv', 'train_source3.tsv']:
            with open(f'student_resource/dataset/train/{src}', 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 4 and p[3] == country:
                        rec = normalize_record(p[0], p[1], p[2], p[3])
                        idx = len(pool)
                        pool.append(rec)
                        pool_id_to_idx[p[0]] = idx
        
        print(f"  Pool: {len(pool):,} records")
        
        # Build index
        name_idx, num_idx, addr_idx = build_index_uncapped(pool, country)
        print(f"  Index: names={len(name_idx):,}, nums={len(num_idx):,}, addrs={len(addr_idx):,}")
        
        # Process each S1
        false_sing = 0
        false_match_sing = 0
        blocking_recall_total = 0
        blocking_recall_found = 0
        
        for i, parts in enumerate(s1_list):
            sid = parts[0]
            s1_rec = normalize_record(parts[0], parts[1], parts[2], parts[3])
            true = gt.get(sid, set())
            
            # Get candidates
            cand_indices = query_candidates(s1_rec, name_idx, num_idx, addr_idx, top_k=50)
            cand_ids = {pool[ci]['entity_id'] for ci in cand_indices}
            
            # Check blocking recall
            true_in_pool = {tid for tid in true if tid in pool_id_to_idx}
            if true_in_pool:
                found = len(true_in_pool & cand_ids)
                blocking_recall_total += len(true_in_pool)
                blocking_recall_found += found
            
            # Run ML model
            if cand_indices:
                batch = []
                for ci in cand_indices:
                    feats = compute_features(s1_rec, pool[ci])
                    batch.append([feats[fn] for fn in FEATURE_NAMES])
                
                X = np.array(batch, dtype=np.float32)
                probs = model.predict(X)
                
                matched = []
                for ci, prob in zip(cand_indices, probs):
                    if prob >= 0.50:  # Lower threshold for testing
                        matched.append(pool[ci]['entity_id'])
                pred = set(matched)
            else:
                pred = set()
            
            # Score
            if len(true) == 0:
                s = 1.0 if len(pred) == 0 else 0.0
                if len(pred) > 0:
                    false_match_sing += 1
            elif len(pred) == 0:
                s = 0.0
                false_sing += 1
                if i < 20 or (len(true) > 0 and len(cand_ids & true) == 0):
                    all_debug.append({
                        'sid': sid, 'name': parts[1][:50], 'addr': parts[2][:50],
                        'true': list(true)[:3], 'cands_found': len(cand_ids & true),
                        'total_cands': len(cand_ids), 'true_in_pool': len(true_in_pool),
                        'score': s, 'issue': 'false_singleton'
                    })
            else:
                tp = len(pred & true)
                fp = len(pred - true)
                fn = len(true - pred)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                s = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
            
            all_scores.append(s)
        
        blocking_recall = blocking_recall_found / blocking_recall_total if blocking_recall_total > 0 else 0
        print(f"  Blocking recall: {blocking_recall:.4f} ({blocking_recall_found:,}/{blocking_recall_total:,})")
        print(f"  False singletons: {false_sing:,}, False matches on singletons: {false_match_sing:,}")
        print(f"  Time: {time.time()-tc:.1f}s")
        
        del pool, name_idx, num_idx, addr_idx
    
    macro = sum(all_scores) / len(all_scores)
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC RESULTS")
    print(f"{'='*70}")
    print(f"  Macro F0.5: {macro:.6f}")
    print(f"  Perfect (>=0.99): {sum(1 for s in all_scores if s >= 0.99):,}/{len(all_scores):,}")
    print(f"  Zero: {sum(1 for s in all_scores if s < 0.01):,}/{len(all_scores):,}")
    
    # Show some debug cases
    blocking_fails = [d for d in all_debug if d.get('issue') == 'false_singleton' and d['cands_found'] == 0]
    if blocking_fails:
        print(f"\n  BLOCKING FAILURES (true matches not found by blocking):")
        for d in blocking_fails[:10]:
            print(f"    {d['sid']}: name='{d['name']}' addr='{d['addr']}'")
            print(f"      True matches: {d['true']}, In pool: {d['true_in_pool']}, "
                  f"Cands found with true: {d['cands_found']}/{d['total_cands']}")
    
    print(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
