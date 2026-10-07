import sys, os, time, pickle, random
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
from src.normalize import normalize_record
from src.features import compute_features, FEATURE_NAMES
from src.evaluate import compute_macro_f05, compute_f05_per_entity
from src.train_ensemble import EnsemblePredictor
import __main__
__main__.EnsemblePredictor = EnsemblePredictor

with open('output/models_ensemble.pkl', 'rb') as f:
    saved = pickle.load(f)
models = saved['models']

# Load ground truth
gt = {}
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        gt[p[0]] = set(p[1].split(',')) if len(p) > 1 and p[1].strip() else set()

random.seed(42)
s1_samples = {'India': [], 'US': []}
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.strip().split('\t')
        c = p[3] if len(p) > 3 else ''
        if c in s1_samples and len(s1_samples[c]) < 500:
            s1_samples[c].append(normalize_record(p[0], p[1], p[2] if len(p)>2 else '', c))
        if len(s1_samples['India']) >= 500 and len(s1_samples['US']) >= 500:
            break

all_s1 = s1_samples['India'] + s1_samples['US']
s1_needed = set()
for r in all_s1:
    s1_needed.update(gt.get(r['entity_id'], set()))

# Load matching pool records + 100,000 distractors
pool_records = []
pool_id_to_idx = {}
for path in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if p[0] in s1_needed or (len(pool_records) < 100000 and random.random() < 0.04):
                idx = len(pool_records)
                rec = normalize_record(p[0], p[1], p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')
                pool_records.append(rec)
                pool_id_to_idx[p[0]] = idx

for c in ['India', 'US']:
    print(f"\nEvaluating Macro F0.5 on {c} (500 entities)...")
    model = models[c]
    s1_list = s1_samples[c]
    
    # For each entity, evaluate true matches + 20 random distractors from same country
    entity_pairs = []
    country_pool = [i for i, r in enumerate(pool_records) if r['country'] == c]
    
    for s1_rec in s1_list:
        true_ids = gt.get(s1_rec['entity_id'], set())
        cands = []
        for tid in true_ids:
            if tid in pool_id_to_idx:
                pi = pool_id_to_idx[tid]
                cands.append((pi, True))
        # Add 15 random distractors
        for _ in range(15):
            r_pi = random.choice(country_pool)
            if pool_records[r_pi]['entity_id'] not in true_ids:
                cands.append((r_pi, False))
                
        # Featurize
        if cands:
            X_rows = []
            for pi, is_t in cands:
                feats = compute_features(s1_rec, pool_records[pi])
                X_rows.append([feats[fn] for fn in FEATURE_NAMES])
            X = np.array(X_rows, dtype=np.float32)
            probs = model.predict(X)
            entity_pairs.append((s1_rec['entity_id'], true_ids, [(cands[i][0], probs[i]) for i in range(len(cands))]))
        else:
            entity_pairs.append((s1_rec['entity_id'], true_ids, []))
            
    for thresh in [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]:
        preds = {}
        for eid, true_ids, cand_probs in entity_pairs:
            matched = {pool_records[pi]['entity_id'] for pi, prob in cand_probs if prob >= thresh}
            preds[eid] = matched
        
        scores = [compute_f05_per_entity(preds[eid], true_ids) for eid, true_ids, _ in entity_pairs]
        macro = sum(scores) / len(scores)
        perfect = sum(1 for s in scores if s >= 0.99)
        zeros = sum(1 for s in scores if s == 0)
        print(f"  Threshold {thresh:.2f}: Macro F0.5 = {macro:.6f} | Perfect: {perfect}/500 | Zeros: {zeros}/500")
