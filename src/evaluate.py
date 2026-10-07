"""
Local F₀.₅ Macro Evaluation — exact replica of the competition scorer.

Computes F₀.₅ per Source 1 entity, then macro-averages.
Singletons (no true matches) score 1.0 if predicted empty, else 0.0.
"""


def compute_f05_per_entity(predicted_ids: set, true_ids: set) -> float:
    """
    Compute F₀.₅ for a single Source 1 entity.

    Args:
        predicted_ids: Set of predicted matched entity IDs (may be empty)
        true_ids:      Set of true matched entity IDs (may be empty)

    Returns:
        F₀.₅ score for this entity (float in [0, 1])
    """
    # Singleton handling
    if len(true_ids) == 0:
        # True singleton: score 1.0 if we correctly predicted empty, else 0.0
        return 1.0 if len(predicted_ids) == 0 else 0.0

    if len(predicted_ids) == 0:
        # We predicted singleton but there ARE true matches → missed everything
        return 0.0

    # Standard precision/recall/F₀.₅
    tp = len(predicted_ids & true_ids)
    fp = len(predicted_ids - true_ids)
    fn = len(true_ids - predicted_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    beta = 0.5
    f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return f_beta


def compute_macro_f05(predictions: dict, ground_truth: dict) -> float:
    """
    Compute macro-averaged F₀.₅ across all Source 1 entities.

    Args:
        predictions:  dict mapping source1_entity_id → set of matched entity IDs
        ground_truth: dict mapping source1_entity_id → set of true matched entity IDs

    Returns:
        Macro F₀.₅ score (float in [0, 1])
    """
    scores = []
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        true = ground_truth[s1_id]
        scores.append(compute_f05_per_entity(pred, true))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def parse_ground_truth(gt_path: str) -> dict:
    """
    Parse ground truth TSV into dict: {source1_entity_id: set(matched_ids)}.
    """
    ground_truth = {}
    with open(gt_path, 'r', encoding='utf-8') as f:
        header = next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                matched = {x.strip() for x in parts[1].split(',') if x.strip()}
            else:
                matched = set()
            ground_truth[s1_id] = matched
    return ground_truth


def parse_predictions(pred_path: str) -> dict:
    """
    Parse predictions TSV into dict: {source1_entity_id: set(matched_ids)}.
    """
    predictions = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        header = next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                matched = {x.strip() for x in parts[1].split(',') if x.strip()}
            else:
                matched = set()
            predictions[s1_id] = matched
    return predictions


def evaluate_from_dicts(predictions: dict, ground_truth: dict,
                        verbose: bool = True) -> float:
    """
    Full evaluation with optional detailed breakdown.
    """
    total = len(ground_truth)
    true_singletons = sum(1 for v in ground_truth.values() if len(v) == 0)
    true_matched = total - true_singletons

    pred_singletons = sum(1 for s1 in ground_truth if len(predictions.get(s1, set())) == 0)

    f05 = compute_macro_f05(predictions, ground_truth)

    if verbose:
        # Detailed breakdown
        singleton_scores = []
        matched_scores = []
        for s1_id, true_ids in ground_truth.items():
            pred_ids = predictions.get(s1_id, set())
            score = compute_f05_per_entity(pred_ids, true_ids)
            if len(true_ids) == 0:
                singleton_scores.append(score)
            else:
                matched_scores.append(score)

        avg_singleton = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0
        avg_matched = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0

        print(f"  Total S1 entities:     {total:,}")
        print(f"  True singletons:       {true_singletons:,} ({100*true_singletons/total:.1f}%)")
        print(f"  True matched:          {true_matched:,} ({100*true_matched/total:.1f}%)")
        print(f"  Predicted singletons:  {pred_singletons:,}")
        print(f"  ─────────────────────────────")
        print(f"  Avg F₀.₅ (singletons): {avg_singleton:.4f}")
        print(f"  Avg F₀.₅ (matched):    {avg_matched:.4f}")
        print(f"  ═══════════════════════════════")
        print(f"  MACRO F₀.₅:            {f05:.4f}")

    return f05


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test with known example from problem statement
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("EVALUATION SELF-TEST")
    print("=" * 60)

    # Example from problem statement
    pred = {'S2-00047', 'S2-00193', 'S3-00812'}
    true = {'S2-00047', 'S3-00812'}
    score = compute_f05_per_entity(pred, true)
    print(f"\nExample from problem statement:")
    print(f"  Predicted: {pred}")
    print(f"  True:      {true}")
    print(f"  F₀.₅ = {score:.3f}  (expected: 0.714)")
    assert abs(score - 0.714) < 0.001, f"Expected ~0.714, got {score}"

    # Singleton: correctly predicted empty
    score_s1 = compute_f05_per_entity(set(), set())
    print(f"\nSingleton (correct empty): F₀.₅ = {score_s1:.3f}  (expected: 1.0)")
    assert score_s1 == 1.0

    # Singleton: wrongly predicted match
    score_s2 = compute_f05_per_entity({'S2-00001'}, set())
    print(f"Singleton (wrong match):   F₀.₅ = {score_s2:.3f}  (expected: 0.0)")
    assert score_s2 == 0.0

    # Macro average test
    preds = {
        'S1-1': {'S2-00047', 'S2-00193', 'S3-00812'},  # F₀.₅ = 0.714
        'S1-2': set(),                                   # true singleton → 1.0
        'S1-3': {'S2-00001'},                            # false singleton → 0.0
    }
    truths = {
        'S1-1': {'S2-00047', 'S3-00812'},
        'S1-2': set(),
        'S1-3': set(),
    }
    macro = compute_macro_f05(preds, truths)
    expected = (0.714 + 1.0 + 0.0) / 3
    print(f"\nMacro F₀.₅ test: {macro:.4f}  (expected: {expected:.4f})")
    assert abs(macro - expected) < 0.001

    print("\n✅ All self-tests passed!")
