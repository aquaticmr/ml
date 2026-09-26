import numpy as np
import pandas as pd
from collections import defaultdict

# ------------------------------------------------------------
# 1. Block Keys Generation
# ------------------------------------------------------------
def create_block_keys(df):
    d = {}
    d["name_exact"] = df["country"].astype(str) + "|" + df["name_norm"].astype(str)
    d["name_core"] = df["country"].astype(str) + "|" + df["name_core"].astype(str)
    d["postal_name4"] = df["country"].astype(str) + "|" + df["postal"].astype(str) + "|" + df["name_prefix4"].astype(str)
    d["postal_name6"] = df["country"].astype(str) + "|" + df["postal"].astype(str) + "|" + df["name_prefix6"].astype(str)
    d["postal_first"] = df["country"].astype(str) + "|" + df["postal"].astype(str) + "|" + df["name_first"].astype(str)
    d["first_last"] = df["country"].astype(str) + "|" + df["name_first"].astype(str) + "|" + df["name_last"].astype(str)
    d["number_first"] = df["country"].astype(str) + "|" + df["numbers"].astype(str) + "|" + df["name_first"].astype(str)
    d["number_name4"] = df["country"].astype(str) + "|" + df["numbers"].astype(str) + "|" + df["name_prefix4"].astype(str)
    
    return pd.DataFrame(d)

# ------------------------------------------------------------
# 2. Block Quality Analysis
# ------------------------------------------------------------
def block_statistics(block_df):
    rows = []
    for col in block_df.columns:
        vc = block_df[col].value_counts()
        vc = vc[vc.index != ""]
        rows.append({
            "rule": col,
            "blocks": len(vc),
            "mean": vc.mean(),
            "median": vc.median(),
            "p90": vc.quantile(.90),
            "p95": vc.quantile(.95),
            "p99": vc.quantile(.99),
            "max": vc.max(),
            "blocks_gt_100": (vc > 100).sum(),
            "blocks_gt_1000": (vc > 1000).sum(),
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# 3. Candidate Generation (with max block size)
# ------------------------------------------------------------
def build_inverted_index(block_series, max_block_size=100):
    temp = pd.DataFrame({
        "key": block_series.values,
        "idx": np.arange(len(block_series), dtype=np.int32)
    })
    temp = temp[temp["key"] != ""]
    
    counts = temp["key"].value_counts()
    valid_keys = counts[counts <= max_block_size].index
    
    temp = temp[temp["key"].isin(valid_keys)]
    groups = temp.groupby("key", sort=False)["idx"].apply(np.asarray).to_dict()
    
    return groups

def generate_candidates(source_df, target_df, source_blocks, target_blocks, rules, max_block_size=100, chunk_size=100_000):
    indexes = {}
    for rule in rules:
        indexes[rule] = build_inverted_index(target_blocks[rule], max_block_size=max_block_size)

    results = []
    source_n = len(source_df)

    for start in range(0, source_n, chunk_size):
        end = min(start + chunk_size, source_n)
        source_chunk = source_df.iloc[start:end]
        candidate_pairs = set()

        for rule in rules:
            target_index = indexes[rule]
            keys = source_blocks.iloc[start:end][rule].values

            for local_i, key in enumerate(keys):
                if not key:
                    continue
                matches = target_index.get(key)
                if matches is None:
                    continue
                
                s1_idx = start + local_i
                for target_idx in matches:
                    candidate_pairs.add((s1_idx, int(target_idx)))

        if candidate_pairs:
            arr = np.asarray(list(candidate_pairs), dtype=np.int64)
            results.append(arr)

        if start % (chunk_size * 10) == 0:
            print(f"{start:,}/{source_n:,}")

    if not results:
        return np.empty((0, 2), dtype=np.int64)

    return np.vstack(results)

# ------------------------------------------------------------
# 4. Recall Calculation
# ------------------------------------------------------------
def calculate_candidate_recall(candidates, ground_truth):
    candidate_set = set(map(tuple, candidates))
    recovered = len(candidate_set.intersection(ground_truth))
    total = len(ground_truth)
    recall = (recovered / total * 100) if total else 0

    return {
        "candidate_pairs": len(candidate_set),
        "true_pairs": total,
        "recovered_pairs": recovered,
        "pair_recall": recall
    }