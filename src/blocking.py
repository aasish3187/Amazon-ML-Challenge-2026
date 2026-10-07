"""
Stage 3: Multi-Strategy Blocking for Entity Resolution.
Generates candidate pairs using TF-IDF + token overlap + numeric blocking,
all unioned for high recall. Works in chunks to fit in 16 GB RAM.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys
import time


def tfidf_blocking(s1_names: list, pool_names: list, pool_ids: list,
                   top_k: int = 20, ngram_range=(2, 4),
                   max_features: int = 80000,
                   chunk_size: int = 2000) -> dict:
    """
    TF-IDF character n-gram blocking.
    For each S1 entity, find top-K most similar candidates from the pool.

    Args:
        s1_names: list of normalized business names for S1 entities
        pool_names: list of normalized names for S2+S3 pool
        pool_ids: list of entity_ids corresponding to pool_names
        top_k: number of top candidates to keep per S1
        ngram_range: character n-gram range for TF-IDF
        max_features: max vocabulary size
        chunk_size: process S1 in chunks of this size

    Returns:
        dict mapping s1_index → set of pool indices
    """
    print(f"    [TF-IDF] Fitting vectorizer on {len(pool_names):,} pool names...")
    t0 = time.time()

    # Fit TF-IDF on pool
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
    )
    pool_matrix = vectorizer.fit_transform(pool_names)
    print(f"    [TF-IDF] Pool matrix: {pool_matrix.shape}, took {time.time()-t0:.1f}s")

    candidates = defaultdict(set)

    # Process S1 in chunks
    n_chunks = (len(s1_names) + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(s1_names))
        chunk_names = s1_names[start:end]

        s1_chunk_matrix = vectorizer.transform(chunk_names)
        sim_matrix = cosine_similarity(s1_chunk_matrix, pool_matrix)

        for i in range(sim_matrix.shape[0]):
            row = sim_matrix[i]
            # Get top-K indices
            if len(row) <= top_k:
                top_indices = np.where(row > 0)[0]
            else:
                top_indices = np.argpartition(row, -top_k)[-top_k:]
                # Filter to only those with positive similarity
                top_indices = top_indices[row[top_indices] > 0.01]

            s1_idx = start + i
            candidates[s1_idx].update(top_indices.tolist())

        if (chunk_idx + 1) % 5 == 0 or chunk_idx == n_chunks - 1:
            print(f"    [TF-IDF] Chunk {chunk_idx+1}/{n_chunks} done "
                  f"({time.time()-t0:.1f}s elapsed)")

    return candidates


def token_overlap_blocking(s1_token_sets: list, pool_token_sets: list,
                           min_overlap: int = 2) -> dict:
    """
    Block pairs that share at least `min_overlap` name tokens.
    Uses inverted index for efficiency.

    Returns:
        dict mapping s1_index → set of pool indices
    """
    print(f"    [TokenOverlap] Building inverted index on {len(pool_token_sets):,} records...")
    t0 = time.time()

    # Build inverted index: token → set of pool indices
    inv_index = defaultdict(set)
    for idx, tokens in enumerate(pool_token_sets):
        for token in tokens:
            if len(token) >= 3:  # Skip very short tokens
                inv_index[token].add(idx)

    print(f"    [TokenOverlap] Index has {len(inv_index):,} unique tokens")

    candidates = defaultdict(set)
    for s1_idx, s1_tokens in enumerate(s1_token_sets):
        # Count how many times each pool record co-occurs
        pool_counts = defaultdict(int)
        for token in s1_tokens:
            if len(token) >= 3 and token in inv_index:
                for pool_idx in inv_index[token]:
                    pool_counts[pool_idx] += 1

        # Keep those with sufficient overlap
        for pool_idx, count in pool_counts.items():
            if count >= min_overlap:
                candidates[s1_idx].add(pool_idx)

    print(f"    [TokenOverlap] Done in {time.time()-t0:.1f}s")
    return candidates


def numeric_blocking(s1_addr_nums: list, pool_addr_nums: list,
                     min_shared: int = 1) -> dict:
    """
    Block pairs that share at least `min_shared` numeric address tokens.
    Catches matches where addresses have the same street/building number.

    Returns:
        dict mapping s1_index → set of pool indices
    """
    print(f"    [NumericBlock] Building numeric index...")
    t0 = time.time()

    # Build inverted index: number → set of pool indices
    num_index = defaultdict(set)
    for idx, nums in enumerate(pool_addr_nums):
        for num in nums:
            if len(num) >= 2:  # Skip single-digit numbers (too common)
                num_index[num].add(idx)

    candidates = defaultdict(set)
    for s1_idx, nums in enumerate(s1_addr_nums):
        if not nums:
            continue
        pool_counts = defaultdict(int)
        for num in nums:
            if len(num) >= 2 and num in num_index:
                for pool_idx in num_index[num]:
                    pool_counts[pool_idx] += 1
        for pool_idx, count in pool_counts.items():
            if count >= min_shared:
                candidates[s1_idx].add(pool_idx)

    print(f"    [NumericBlock] Done in {time.time()-t0:.1f}s")
    return candidates


def first_n_char_blocking(s1_names: list, pool_names: list, n: int = 5) -> dict:
    """
    Block pairs whose normalized names share the same first N characters.

    Returns:
        dict mapping s1_index → set of pool indices
    """
    print(f"    [FirstNChar] Building prefix index (n={n})...")
    t0 = time.time()

    prefix_index = defaultdict(set)
    for idx, name in enumerate(pool_names):
        if len(name) >= n:
            prefix_index[name[:n]].add(idx)

    candidates = defaultdict(set)
    for s1_idx, name in enumerate(s1_names):
        if len(name) >= n:
            prefix = name[:n]
            if prefix in prefix_index:
                candidates[s1_idx].update(prefix_index[prefix])

    print(f"    [FirstNChar] Done in {time.time()-t0:.1f}s")
    return candidates


def union_blocking_results(*blocking_dicts) -> dict:
    """Union multiple blocking results into one."""
    merged = defaultdict(set)
    for bd in blocking_dicts:
        for s1_idx, pool_indices in bd.items():
            merged[s1_idx].update(pool_indices)
    return merged


def run_full_blocking(s1_records: list, pool_records: list,
                      top_k: int = 20, max_candidates: int = 50) -> dict:
    """
    Run all blocking strategies and union results.

    Args:
        s1_records: list of normalized S1 record dicts
        pool_records: list of normalized S2+S3 record dicts
        top_k: TF-IDF top-K
        max_candidates: safety cap per S1 entity

    Returns:
        dict mapping s1_index → set of pool_indices
    """
    print(f"\n  Running multi-strategy blocking: {len(s1_records):,} S1 × {len(pool_records):,} pool")

    s1_names = [r['norm_name'] for r in s1_records]
    pool_names = [r['norm_name'] for r in pool_records]
    s1_addrs = [r['norm_addr'] for r in s1_records]
    pool_addrs = [r['norm_addr'] for r in pool_records]
    s1_tokens = [r['name_tokens'] for r in s1_records]
    pool_tokens = [r['name_tokens'] for r in pool_records]
    s1_nums = [r['addr_numbers'] for r in s1_records]
    pool_nums = [r['addr_numbers'] for r in pool_records]

    # Combined name + address
    s1_combined = [f"{r['norm_name']} {r['norm_addr']}".strip() for r in s1_records]
    pool_combined = [f"{r['norm_name']} {r['norm_addr']}".strip() for r in pool_records]

    # Strategy 1: Name TF-IDF (char 2-4 n-grams)
    name_tfidf_cands = tfidf_blocking(s1_names, pool_names,
                                      [r['entity_id'] for r in pool_records],
                                      top_k=top_k, ngram_range=(2, 4))

    # Strategy 2: Address TF-IDF (char 3-5 n-grams)
    addr_tfidf_cands = tfidf_blocking(s1_addrs, pool_addrs,
                                      [r['entity_id'] for r in pool_records],
                                      top_k=min(top_k, 25), ngram_range=(3, 5))

    # Strategy 3: Combined (Name + Address) TF-IDF
    comb_tfidf_cands = tfidf_blocking(s1_combined, pool_combined,
                                      [r['entity_id'] for r in pool_records],
                                      top_k=min(top_k, 25), ngram_range=(3, 5))

    # Strategy 4: Token overlap blocking on name (≥2 shared tokens)
    token_cands = token_overlap_blocking(s1_tokens, pool_tokens, min_overlap=2)

    # Strategy 5: Numeric address blocking (same street/building number)
    numeric_cands = numeric_blocking(s1_nums, pool_nums, min_shared=1)

    # Strategy 6: First-5-character prefix blocking
    prefix_cands = first_n_char_blocking(s1_names, pool_names, n=5)

    # Union with priority: TF-IDF first, then token overlap, then prefix, then numeric
    all_cands = defaultdict(set)
    for s1_idx in range(len(s1_records)):
        ordered_cands = []
        # Add high-priority candidates in order
        for source in [comb_tfidf_cands, name_tfidf_cands, addr_tfidf_cands,
                       token_cands, prefix_cands, numeric_cands]:
            for pool_idx in source.get(s1_idx, ()):
                if pool_idx not in ordered_cands:
                    ordered_cands.append(pool_idx)
                if len(ordered_cands) >= max_candidates:
                    break
            if len(ordered_cands) >= max_candidates:
                break
        all_cands[s1_idx] = set(ordered_cands)

    # Stats
    n_with_cands = sum(1 for v in all_cands.values() if len(v) > 0)
    total_pairs = sum(len(v) for v in all_cands.values())
    avg_cands = total_pairs / len(s1_records) if s1_records else 0

    print(f"\n  Blocking Summary:")
    print(f"    S1 with candidates: {n_with_cands:,} / {len(s1_records):,}")
    print(f"    Total candidate pairs: {total_pairs:,}")
    print(f"    Avg candidates per S1: {avg_cands:.1f}")

    return all_cands


if __name__ == '__main__':
    print("Blocking module loaded. Use run_full_blocking() from pipeline.")
