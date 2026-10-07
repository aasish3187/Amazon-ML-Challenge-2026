"""
Stage 4: Pairwise Feature Engineering for Entity Resolution.
Computes 18 discriminative features for each (S1, candidate) pair.
"""
from rapidfuzz import fuzz, distance


def _safe_str(val) -> str:
    """Convert to string, handling NaN/None."""
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return ''
    return str(val)


def _jaccard_tokens(set_a: set, set_b: set) -> float:
    """Token-level Jaccard similarity."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _common_token_ratio(set_a: set, set_b: set) -> float:
    """Fraction of tokens in common relative to the smaller set."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _numeric_jaccard(nums_a: list, nums_b: list) -> float:
    """Jaccard on numeric tokens (street numbers, PIN codes)."""
    if not nums_a and not nums_b:
        return 0.0
    set_a = set(nums_a)
    set_b = set(nums_b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _numeric_exact_match(nums_a: list, nums_b: list) -> float:
    """Do the numeric tokens match exactly?"""
    if not nums_a and not nums_b:
        return 0.0  # No information
    if not nums_a or not nums_b:
        return 0.0
    return 1.0 if set(nums_a) == set(nums_b) else 0.0


def compute_features(s1_record: dict, candidate_record: dict) -> dict:
    """
    Compute all pairwise features for an (S1, candidate) pair.

    Both records should be dicts with keys from normalize.normalize_record():
        - entity_id, norm_name, norm_addr, name_tokens, addr_numbers, country

    Returns: dict of feature_name → float
    """
    s1_name = s1_record.get('norm_name', '')
    c_name = candidate_record.get('norm_name', '')
    s1_addr = s1_record.get('norm_addr', '')
    c_addr = candidate_record.get('norm_addr', '')
    s1_name_tok = s1_record.get('name_tokens', set())
    c_name_tok = candidate_record.get('name_tokens', set())
    s1_addr_nums = s1_record.get('addr_numbers', [])
    c_addr_nums = candidate_record.get('addr_numbers', [])

    features = {}

    # ═══ NAME FEATURES (8) ═══════════════════════════════════════════════════

    # 1. Jaro-Winkler similarity (favors prefix matches)
    features['name_jaro_winkler'] = (
        distance.JaroWinkler.similarity(s1_name, c_name) if s1_name and c_name else 0.0
    )

    # 2. Levenshtein ratio (normalized edit distance)
    features['name_levenshtein_ratio'] = (
        fuzz.ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    )

    # 3. Token sort ratio (word-order invariant)
    features['name_token_sort_ratio'] = (
        fuzz.token_sort_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    )

    # 4. Token set ratio (handles subset/superset relationships)
    features['name_token_set_ratio'] = (
        fuzz.token_set_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    )

    # 5. Partial ratio (best substring match)
    features['name_partial_ratio'] = (
        fuzz.partial_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    )

    # 6. Token Jaccard similarity
    features['name_jaccard'] = _jaccard_tokens(s1_name_tok, c_name_tok)

    # 7. Common token ratio (fraction of shared tokens)
    features['name_common_token_ratio'] = _common_token_ratio(s1_name_tok, c_name_tok)

    # 8. Length difference (catches garbage/very different entities)
    features['name_len_diff'] = abs(len(s1_name) - len(c_name))

    # ═══ ADDRESS FEATURES (7) ════════════════════════════════════════════════

    # 9. Address token sort ratio
    features['addr_token_sort_ratio'] = (
        fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    )

    # 10. Address Jaccard
    s1_addr_tok = set(s1_addr.split()) if s1_addr else set()
    c_addr_tok = set(c_addr.split()) if c_addr else set()
    features['addr_jaccard'] = _jaccard_tokens(s1_addr_tok, c_addr_tok)

    # 11. Address Levenshtein ratio
    features['addr_levenshtein_ratio'] = (
        fuzz.ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    )

    # 12. Numeric token match (street numbers, PIN codes)
    features['addr_number_match'] = _numeric_exact_match(s1_addr_nums, c_addr_nums)

    # 13. Numeric Jaccard
    features['addr_number_jaccard'] = _numeric_jaccard(s1_addr_nums, c_addr_nums)

    # 14. Has address info (at least one side has non-empty address)
    features['addr_has_any'] = 1.0 if (s1_addr or c_addr) else 0.0

    # 15. Both addresses empty (cannot discriminate on address)
    features['addr_both_empty'] = 1.0 if (not s1_addr and not c_addr) else 0.0

    # ═══ CROSS FEATURES (3) ══════════════════════════════════════════════════

    # 16. Weighted combined score
    features['combined_score'] = (
        0.6 * features['name_token_sort_ratio'] +
        0.4 * features['addr_jaccard']
    )

    # 17. Name × Address interaction (high only when BOTH are similar)
    features['name_addr_product'] = (
        features['name_jaro_winkler'] * features['addr_jaccard']
    )

    # ═══ HIGH-PRECISION DISCRIMINATIVE FEATURES (7) ═════════════════════════

    # 19. Exact name and address match indicators
    features['name_exact_match'] = 1.0 if (s1_name and s1_name == c_name) else 0.0
    features['addr_exact_match'] = 1.0 if (s1_addr and s1_addr == c_addr) else 0.0

    # 20. First token (brand anchor) match and ratio
    s1_tokens = [t for t in s1_name.split() if len(t) >= 2]
    c_tokens = [t for t in c_name.split() if len(t) >= 2]
    if s1_tokens and c_tokens:
        features['first_token_ratio'] = fuzz.ratio(s1_tokens[0], c_tokens[0]) / 100.0
        features['first_token_match'] = 1.0 if s1_tokens[0] == c_tokens[0] else 0.0
    else:
        features['first_token_ratio'] = 0.0
        features['first_token_match'] = 0.0

    # 21. Postal / PIN code match and conflict
    # In India PIN = 6 digits; in US ZIP = 5 digits
    p1 = next((n for n in s1_addr_nums if len(n) == 6 and n[0] != '0' or len(n) == 5), '')
    p2 = next((n for n in c_addr_nums if len(n) == 6 and n[0] != '0' or len(n) == 5), '')
    if p1 and p2:
        features['pincode_match'] = 1.0 if p1 == p2 else 0.0
        features['pincode_conflict'] = 1.0 if p1 != p2 else 0.0
    else:
        features['pincode_match'] = 0.0
        features['pincode_conflict'] = 0.0

    # 22. Acronym / Initials match
    init_1 = "".join([t[0] for t in s1_tokens]) if s1_tokens else ""
    init_2 = "".join([t[0] for t in c_tokens]) if c_tokens else ""
    features['initials_match'] = 1.0 if (len(init_1) >= 2 and init_1 == init_2) else 0.0

    # 23. Substring Containment Ratio (name)
    if s1_name and c_name:
        if s1_name in c_name or c_name in s1_name:
            features['name_containment_ratio'] = min(len(s1_name), len(c_name)) / max(len(s1_name), len(c_name))
        else:
            features['name_containment_ratio'] = 0.0
    else:
        features['name_containment_ratio'] = 0.0

    # 24. Character 3-gram Jaccard (robust against typos & transliterations)
    s1_clean = s1_name.replace(' ', '')
    c_clean = c_name.replace(' ', '')
    if len(s1_clean) >= 3 and len(c_clean) >= 3:
        g1 = {s1_clean[i:i+3] for i in range(len(s1_clean)-2)}
        g2 = {c_clean[i:i+3] for i in range(len(c_clean)-2)}
        features['name_char_3gram_jaccard'] = len(g1 & g2) / len(g1 | g2) if (g1 | g2) else 0.0
    else:
        features['name_char_3gram_jaccard'] = 0.0

    # 25. Substring Containment Ratio (address)
    if s1_addr and c_addr:
        if s1_addr in c_addr or c_addr in s1_addr:
            features['addr_containment_ratio'] = min(len(s1_addr), len(c_addr)) / max(len(s1_addr), len(c_addr))
        else:
            features['addr_containment_ratio'] = 0.0
    else:
        features['addr_containment_ratio'] = 0.0

    # 18. Country exact match (sanity — should always be 1 with partitioning)
    s1_country = _safe_str(s1_record.get('country', ''))
    c_country = _safe_str(candidate_record.get('country', ''))
    features['country_match'] = 1.0 if s1_country == c_country else 0.0

    return features


FEATURE_NAMES = [
    'name_jaro_winkler', 'name_levenshtein_ratio', 'name_token_sort_ratio',
    'name_token_set_ratio', 'name_partial_ratio', 'name_jaccard',
    'name_common_token_ratio', 'name_len_diff',
    'addr_token_sort_ratio', 'addr_jaccard', 'addr_levenshtein_ratio',
    'addr_number_match', 'addr_number_jaccard', 'addr_has_any', 'addr_both_empty',
    'combined_score', 'name_addr_product', 'country_match',
    'name_exact_match', 'addr_exact_match', 'first_token_ratio', 'first_token_match',
    'pincode_match', 'pincode_conflict', 'initials_match',
]



# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    sys.stdout.reconfigure(encoding='utf-8')
    from src.normalize import normalize_record

    print("=" * 70)
    print("FEATURE ENGINEERING SELF-TEST")
    print("=" * 70)

    s1 = normalize_record("S1-001", "Moran Staffing LLC",
                          "1902 Shamrock Road, Dothan, AL", "US")
    c1 = normalize_record("S3-001", "Moran Seaffng LLC",
                          "1902 Shamrock Rd, Dothan, Alabama", "US")
    c2 = normalize_record("S2-999", "Totally Different Corp",
                          "999 Other Street, NYC, NY", "US")

    feats_match = compute_features(s1, c1)
    feats_nomatch = compute_features(s1, c2)

    print("\n  TRUE MATCH PAIR:")
    for k in FEATURE_NAMES:
        print(f"    {k:30s} = {feats_match[k]:.4f}")

    print("\n  NON-MATCH PAIR:")
    for k in FEATURE_NAMES:
        print(f"    {k:30s} = {feats_nomatch[k]:.4f}")

    # Sanity: match features should be higher
    assert feats_match['name_token_sort_ratio'] > feats_nomatch['name_token_sort_ratio']
    assert feats_match['addr_number_match'] > feats_nomatch['addr_number_match']
    print("\n✅ Features look correct — match pair scores higher than non-match!")
