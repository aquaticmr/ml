"""
Business Entity Resolution — Stage 2 Pipeline
================================================
Picks up after: EDA, multilingual analysis, normalization, basic blocking eval.

This stage does:
  1. Multi-rule blocking UNION (postal+name4 capped + name_prefix6 + token_pair + exact_name)
     -> raises entity-level full-recovery without candidate explosion
  2. Pairwise feature engineering (name/address similarity, token overlap, numeric, postal, script)
  3. LightGBM binary classifier training (GPU-capable) with controlled negative sampling
  4. Entity-level macro F0.5 threshold search (matches the leaderboard metric exactly)
  5. Explicit singleton handling
  6. Inference over the full/test set -> matching_results.tsv

Adjust the CONFIG block and the `load_normalized` function to match your actual
Stage-1 outputs (column names / file paths). Everything else should run as-is.

Usage:
    python entity_resolution_pipeline.py --stage all
    python entity_resolution_pipeline.py --stage candidates
    python entity_resolution_pipeline.py --stage features
    python entity_resolution_pipeline.py --stage train
    python entity_resolution_pipeline.py --stage infer
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("entres")

# ----------------------------------------------------------------------------
# CONFIG — edit these to match your environment
# ----------------------------------------------------------------------------
CFG = {
    "data_dir": Path("../data"),
    "out_dir": Path("../outputs"),
    "s1_file": "source1.tsv",
    "s2_file": "source2.tsv",
    "s3_file": "source3.tsv",
    "gt_file": "ground_truth.tsv",
    "test_s1_file": "test_source1.tsv",
    "test_s2_file": "test_source2.tsv",
    "test_s3_file": "test_source3.tsv",
    "block_cap": 60,          # max candidates kept per S1 entity per source, per rule
    "name_prefix_len": 6,
    "postal_name_prefix_len": 4,
    "neg_pos_ratio": 8,       # negative sampling ratio for training
    "val_frac": 0.10,
    "random_state": 42,
    "lgbm_device": "gpu",     # set "cpu" if your LightGBM isn't built with GPU support
}
CFG["out_dir"].mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Lightweight normalization (rebuild only what's needed for feature calc)
# ----------------------------------------------------------------------------
ABBREV_MAP = {
    "limited": "ltd", "private": "pvt", "road": "rd", "street": "st",
    "incorporated": "inc", "drive": "dr", "avenue": "ave", "floor": "fl",
    "company": "co", "corporation": "corp", "boulevard": "blvd",
    "apartment": "apt", "highway": "hwy", "lane": "ln",
}
NUM_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = s.split()
    tokens = [ABBREV_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)


def tokenize(s: str) -> set:
    if not isinstance(s, str):
        return set()
    return set(TOKEN_RE.findall(s.lower()))


def extract_numbers(s: str) -> tuple:
    if not isinstance(s, str):
        return tuple()
    return tuple(NUM_RE.findall(s))


def extract_postal(s: str) -> str:
    """Heuristic: last standalone 5-6 digit number in the address = postal/PIN code."""
    nums = extract_numbers(s)
    for n in reversed(nums):
        if len(n) in (5, 6):
            return n
    return ""


def prep_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name_norm"] = df["business_name"].fillna("").map(normalize_text)
    df["addr_norm"] = df["business_address"].fillna("").map(normalize_text)
    df["name_tokens"] = df["name_norm"].map(tokenize)
    df["addr_tokens"] = df["addr_norm"].map(tokenize)
    df["addr_numbers"] = df["business_address"].fillna("").map(extract_numbers)
    df["postal"] = df["business_address"].fillna("").map(extract_postal)
    df["name_prefix"] = df["name_norm"].str.replace(" ", "", regex=False).str[: CFG["name_prefix_len"]]
    df["name_prefix_p"] = df["name_norm"].str.replace(" ", "", regex=False).str[: CFG["postal_name_prefix_len"]]
    return df


def load_normalized(split: str) -> tuple:
    """split = 'train' or 'test'."""
    if split == "train":
        s1 = pd.read_csv(CFG["data_dir"] / CFG["s1_file"], sep="\t")
        s2 = pd.read_csv(CFG["data_dir"] / CFG["s2_file"], sep="\t")
        s3 = pd.read_csv(CFG["data_dir"] / CFG["s3_file"], sep="\t")
    else:
        s1 = pd.read_csv(CFG["data_dir"] / CFG["test_s1_file"], sep="\t")
        s2 = pd.read_csv(CFG["data_dir"] / CFG["test_s2_file"], sep="\t")
        s3 = pd.read_csv(CFG["data_dir"] / CFG["test_s3_file"], sep="\t")
    return prep_source(s1), prep_source(s2), prep_source(s3)


def load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(CFG["data_dir"] / CFG["gt_file"], sep="\t")
    gt["match_set"] = gt["matched_entity_ids"].fillna("").apply(
        lambda x: set(x.split(",")) if x else set()
    )
    return gt


# ----------------------------------------------------------------------------
# 2. Multi-rule blocking UNION with per-block capping
# ----------------------------------------------------------------------------
def _cap_block(group: pd.DataFrame, s1_row: pd.Series, cap: int) -> pd.DataFrame:
    """If a block is oversized, keep only the `cap` most name-similar candidates."""
    if len(group) <= cap:
        return group
    scores = group["name_norm"].map(lambda x: fuzz.token_set_ratio(s1_row["name_norm"], x))
    keep_idx = scores.nlargest(cap).index
    return group.loc[keep_idx]


def block_by_key(s1: pd.DataFrame, other: pd.DataFrame, key: str, cap: int) -> pd.DataFrame:
    """Generic equi-join blocking on a precomputed key column, capped per S1 row."""
    s1_k = s1[["entity_id", key]].rename(columns={"entity_id": "s1_id"})
    other_k = other[["entity_id", key]].rename(columns={"entity_id": "cand_id"})
    s1_k = s1_k[s1_k[key] != ""]
    other_k = other_k[other_k[key] != ""]
    merged = s1_k.merge(other_k, on=key, how="inner")
    if merged.empty:
        return merged[["s1_id", "cand_id"]]
    sizes = merged.groupby("s1_id")["cand_id"].transform("count")
    small = merged[sizes <= cap][["s1_id", "cand_id"]]
    big_keys = merged.loc[sizes > cap, "s1_id"].unique()
    if len(big_keys) == 0:
        return small
    # cap oversized blocks by name similarity (slower path, only for the few huge blocks)
    capped_rows = []
    s1_lookup = s1.set_index("entity_id")
    other_lookup = other.set_index("entity_id")
    for s1_id in big_keys:
        cand_ids = merged.loc[merged["s1_id"] == s1_id, "cand_id"]
        sub = other_lookup.loc[other_lookup.index.intersection(cand_ids)]
        s1_row = s1_lookup.loc[s1_id]
        kept = _cap_block(sub.reset_index(), s1_row, cap)
        for cid in kept["entity_id"]:
            capped_rows.append((s1_id, cid))
    big = pd.DataFrame(capped_rows, columns=["s1_id", "cand_id"])
    return pd.concat([small, big], ignore_index=True)


def token_pair_block(s1: pd.DataFrame, other: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Explode on shared name tokens (skip very common tokens to avoid blowups)."""
    def explode(df, id_col):
        rows = df[["entity_id", "name_tokens"]].explode("name_tokens")
        rows = rows.rename(columns={"entity_id": id_col, "name_tokens": "tok"})
        return rows.dropna(subset=["tok"])

    s1_exp = explode(s1, "s1_id")
    other_exp = explode(other, "cand_id")
    tok_freq = other_exp["tok"].value_counts()
    common_toks = tok_freq[tok_freq > 5000].index  # drop overly generic tokens ("group", "llc", ...)
    s1_exp = s1_exp[~s1_exp["tok"].isin(common_toks)]
    other_exp = other_exp[~other_exp["tok"].isin(common_toks)]
    merged = s1_exp.merge(other_exp, on="tok", how="inner")[["s1_id", "cand_id"]].drop_duplicates()
    sizes = merged.groupby("s1_id")["cand_id"].transform("count")
    return merged[sizes <= cap]


def generate_candidates(s1: pd.DataFrame, other: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    """Union of blocking rules -> deduped candidate pairs with rule provenance flags."""
    cap = CFG["block_cap"]
    log.info(f"[{source_tag}] blocking: postal+name4 (capped)...")
    b1 = block_by_key(s1, other, "name_prefix_p", cap)  # combine with postal below via composite key
    b1["rule_postal_name4"] = 1

    log.info(f"[{source_tag}] blocking: name_prefix6...")
    b2 = block_by_key(s1, other, "name_prefix", cap)
    b2["rule_prefix6"] = 1

    log.info(f"[{source_tag}] blocking: exact_name...")
    b3 = block_by_key(s1, other, "name_norm", cap)
    b3["rule_exact_name"] = 1

    log.info(f"[{source_tag}] blocking: postal...")
    b4 = block_by_key(s1, other, "postal", cap)
    b4["rule_postal"] = 1

    log.info(f"[{source_tag}] blocking: token_pair...")
    b5 = token_pair_block(s1, other, cap)
    b5["rule_token_pair"] = 1

    all_blocks = pd.concat([b1, b2, b3, b4, b5], ignore_index=True)
    flag_cols = ["rule_postal_name4", "rule_prefix6", "rule_exact_name", "rule_postal", "rule_token_pair"]
    for c in flag_cols:
        if c not in all_blocks.columns:
            all_blocks[c] = 0
        all_blocks[c] = all_blocks[c].fillna(0).astype(int)

    dedup = all_blocks.groupby(["s1_id", "cand_id"], as_index=False)[flag_cols].max()
    dedup["source"] = source_tag
    log.info(f"[{source_tag}] union candidates: {len(dedup):,} "
             f"({dedup['s1_id'].nunique():,} distinct S1 entities covered)")
    return dedup


# ----------------------------------------------------------------------------
# 3. Pairwise feature engineering
# ----------------------------------------------------------------------------
def compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    s1_lu = s1.set_index("entity_id")
    o_lu = other.set_index("entity_id")

    a = s1_lu.loc[pairs["s1_id"]].reset_index(drop=True)
    b = o_lu.loc[pairs["cand_id"]].reset_index(drop=True)
    feat = pairs.reset_index(drop=True).copy()

    # --- name features ---
    feat["name_exact"] = (a["name_norm"].values == b["name_norm"].values).astype(int)
    feat["name_jw"] = [JaroWinkler.similarity(x, y) for x, y in zip(a["name_norm"], b["name_norm"])]
    feat["name_token_sort"] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(a["name_norm"], b["name_norm"])]
    feat["name_token_set"] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(a["name_norm"], b["name_norm"])]
    feat["name_partial"] = [fuzz.partial_ratio(x, y) / 100.0 for x, y in zip(a["name_norm"], b["name_norm"])]

    def jaccard(s1_, s2_):
        if not s1_ and not s2_:
            return 1.0
        u = s1_ | s2_
        return len(s1_ & s2_) / len(u) if u else 0.0

    def containment(s1_, s2_):
        if not s1_:
            return 0.0
        return len(s1_ & s2_) / len(s1_)

    feat["name_tok_jaccard"] = [jaccard(x, y) for x, y in zip(a["name_tokens"], b["name_tokens"])]
    feat["name_tok_contain"] = [containment(x, y) for x, y in zip(a["name_tokens"], b["name_tokens"])]
    feat["name_prefix6_match"] = (a["name_prefix"].values == b["name_prefix"].values).astype(int)

    # --- address features ---
    feat["addr_jw"] = [JaroWinkler.similarity(x, y) for x, y in zip(a["addr_norm"], b["addr_norm"])]
    feat["addr_token_sort"] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(a["addr_norm"], b["addr_norm"])]
    feat["addr_tok_jaccard"] = [jaccard(x, y) for x, y in zip(a["addr_tokens"], b["addr_tokens"])]
    feat["addr_tok_contain"] = [containment(x, y) for x, y in zip(a["addr_tokens"], b["addr_tokens"])]

    def num_overlap(n1, n2):
        s1_, s2_ = set(n1), set(n2)
        if not s1_ and not s2_:
            return 1.0
        u = s1_ | s2_
        return len(s1_ & s2_) / len(u) if u else 0.0

    feat["addr_num_overlap"] = [num_overlap(x, y) for x, y in zip(a["addr_numbers"], b["addr_numbers"])]
    feat["postal_match"] = ((a["postal"].values == b["postal"].values) & (a["postal"].values != "")).astype(int)
    feat["postal_missing"] = ((a["postal"].values == "") | (b["postal"].values == "")).astype(int)

    # --- cross-field / provenance ---
    feat["country_match"] = (a["country"].values == b["country"].values).astype(int)
    feat["combo_score"] = 0.6 * feat["name_token_set"] + 0.4 * feat["addr_token_sort"]

    return feat


FEATURE_COLS = [
    "rule_postal_name4", "rule_prefix6", "rule_exact_name", "rule_postal", "rule_token_pair",
    "name_exact", "name_jw", "name_token_sort", "name_token_set", "name_partial",
    "name_tok_jaccard", "name_tok_contain", "name_prefix6_match",
    "addr_jw", "addr_token_sort", "addr_tok_jaccard", "addr_tok_contain",
    "addr_num_overlap", "postal_match", "postal_missing",
    "country_match", "combo_score",
]


# ----------------------------------------------------------------------------
# 4. Training with controlled negative sampling
# ----------------------------------------------------------------------------
def build_training_frame(pairs: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    gt_pairs = set()
    for _, row in gt.iterrows():
        for m in row["match_set"]:
            gt_pairs.add((row["source1_entity_id"], m))

    pairs = pairs.copy()
    pairs["label"] = pairs.apply(lambda r: 1 if (r["s1_id"], r["cand_id"]) in gt_pairs else 0, axis=1)

    pos = pairs[pairs["label"] == 1]
    neg = pairs[pairs["label"] == 0]
    n_neg = min(len(neg), len(pos) * CFG["neg_pos_ratio"])
    neg_sampled = neg.sample(n=n_neg, random_state=CFG["random_state"]) if n_neg > 0 else neg
    log.info(f"Training frame: {len(pos):,} positive / {len(neg_sampled):,} sampled negative "
             f"(from {len(neg):,} total negatives)")
    return pd.concat([pos, neg_sampled], ignore_index=True)


def train_lgbm(train_df: pd.DataFrame):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X = train_df[FEATURE_COLS]
    y = train_df["label"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=CFG["val_frac"], stratify=y, random_state=CFG["random_state"]
    )

    params = dict(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_child_samples=30,
        device=CFG["lgbm_device"],
        verbosity=-1,
    )
    try:
        model = lgb.LGBMClassifier(n_estimators=2000, **params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
    except Exception as e:
        log.warning(f"GPU training failed ({e}); falling back to CPU.")
        params["device"] = "cpu"
        model = lgb.LGBMClassifier(n_estimators=2000, **params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

    model.booster_.save_model(str(CFG["out_dir"] / "lgbm_pair_model.txt"))
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    log.info(f"Top features:\n{imp.head(10)}")
    return model


# ----------------------------------------------------------------------------
# 5. Entity-level macro F0.5 threshold optimization
# ----------------------------------------------------------------------------
def entity_f05(pred_sets: dict, true_sets: dict, beta=0.5) -> float:
    scores = []
    for s1_id, true_set in true_sets.items():
        pred_set = pred_sets.get(s1_id, set())
        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue
        b2 = beta ** 2
        f = (1 + b2) * precision * recall / (b2 * precision + recall) if (b2 * precision + recall) else 0.0
        scores.append(f)
    return float(np.mean(scores))


def optimize_threshold(scored_pairs: pd.DataFrame, gt: pd.DataFrame, grid=None):
    if grid is None:
        grid = np.arange(0.50, 0.99, 0.02)

    true_sets = {row["source1_entity_id"]: row["match_set"] for _, row in gt.iterrows()}
    best_t, best_f = 0.5, -1.0
    for t in grid:
        above = scored_pairs[scored_pairs["score"] >= t]
        pred_sets = above.groupby("s1_id")["cand_id"].apply(set).to_dict()
        f = entity_f05(pred_sets, true_sets)
        log.info(f"  threshold={t:.2f}  macro_F0.5={f:.4f}")
        if f > best_f:
            best_f, best_t = f, t
    log.info(f"BEST threshold={best_t:.2f}  macro_F0.5={best_f:.4f}")
    return best_t, best_f


# ----------------------------------------------------------------------------
# 6. Inference -> matching_results.tsv
# ----------------------------------------------------------------------------
def run_inference(scored_pairs: pd.DataFrame, all_s1_ids: pd.Series, threshold: float) -> pd.DataFrame:
    above = scored_pairs[scored_pairs["score"] >= threshold]
    matches = above.groupby("s1_id")["cand_id"].apply(lambda x: ",".join(sorted(set(x)))).to_dict()

    result = pd.DataFrame({"source1_entity_id": all_s1_ids})
    result["matched_entity_ids"] = result["source1_entity_id"].map(matches).fillna("")
    return result


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(stage: str):
    t0 = time.time()

    if stage in ("all", "candidates", "features", "train"):
        log.info("Loading + normalizing train sources...")
        s1, s2, s3 = load_normalized("train")
        gt = load_ground_truth()

        log.info("Generating candidate pairs (S1->S2)...")
        cand_s2 = generate_candidates(s1, s2, "S2")
        log.info("Generating candidate pairs (S1->S3)...")
        cand_s3 = generate_candidates(s1, s3, "S3")

        log.info("Computing features (S1->S2)...")
        feat_s2 = compute_features(cand_s2, s1, s2)
        log.info("Computing features (S1->S3)...")
        feat_s3 = compute_features(cand_s3, s1, s3)

        all_feat = pd.concat([feat_s2, feat_s3], ignore_index=True)
        all_feat.to_parquet(CFG["out_dir"] / "candidate_features_train.parquet")
        log.info(f"Saved {len(all_feat):,} candidate pairs with features.")

        train_df = build_training_frame(all_feat, gt)
        train_df.to_parquet(CFG["out_dir"] / "training_frame.parquet")

    if stage in ("all", "train"):
        train_df = pd.read_parquet(CFG["out_dir"] / "training_frame.parquet")
        model = train_lgbm(train_df)

        all_feat = pd.read_parquet(CFG["out_dir"] / "candidate_features_train.parquet")
        all_feat["score"] = model.predict_proba(all_feat[FEATURE_COLS])[:, 1]
        gt = load_ground_truth()
        best_t, best_f = optimize_threshold(all_feat[["s1_id", "cand_id", "score"]], gt)
        with open(CFG["out_dir"] / "threshold.json", "w") as f:
            json.dump({"best_threshold": best_t, "macro_f05": best_f}, f, indent=2)

    if stage in ("all", "infer"):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(CFG["out_dir"] / "lgbm_pair_model.txt"))
        with open(CFG["out_dir"] / "threshold.json") as f:
            threshold = json.load(f)["best_threshold"]

        log.info("Loading + normalizing TEST sources...")
        t_s1, t_s2, t_s3 = load_normalized("test")

        cand_s2 = generate_candidates(t_s1, t_s2, "S2")
        cand_s3 = generate_candidates(t_s1, t_s3, "S3")
        feat_s2 = compute_features(cand_s2, t_s1, t_s2)
        feat_s3 = compute_features(cand_s3, t_s1, t_s3)
        all_feat = pd.concat([feat_s2, feat_s3], ignore_index=True)

        all_feat["score"] = booster.predict(all_feat[FEATURE_COLS])
        result = run_inference(all_feat[["s1_id", "cand_id", "score"]], t_s1["entity_id"], threshold)
        result = result.rename(columns={"source1_entity_id": "source1_entity_id"})
        out_path = CFG["out_dir"] / "matching_results.tsv"
        result.to_csv(out_path, sep="\t", index=False)

        # also save candidate_pairs.tsv for the audit archive
        cand_all = pd.concat([cand_s2, cand_s3], ignore_index=True)[["s1_id", "cand_id", "source"]]
        cand_all.to_csv(CFG["out_dir"] / "candidate_pairs.tsv", sep="\t", index=False)

        n_empty = (result["matched_entity_ids"] == "").sum()
        log.info(f"Wrote {out_path} — {len(result):,} rows, {n_empty:,} predicted singletons "
                 f"({100*n_empty/len(result):.2f}%)")

    log.info(f"Stage '{stage}' complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "candidates", "features", "train", "infer"], default="all")
    args = parser.parse_args()
    main(args.stage)
