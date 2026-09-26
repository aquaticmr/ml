# ============================================================
# BUSINESS ENTITY RESOLUTION
# STAGE 2:
# Candidate retrieval -> feature engineering ->
# hard negative sampling -> CatBoost -> validation -> test
# ============================================================

import os
import re
import gc
import math
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rapidfuzz.fuzz import (
    ratio,
    WRatio,
    token_sort_ratio,
    token_set_ratio
)

from catboost import CatBoostClassifier, Pool


# ============================================================
# CONFIG & PATH RESOLUTION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

DATA_DIR = os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "model"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_data_file(preferred_dir, *names):
    """Find dataset file across preferred directory and student_resource fallback."""
    for name in names:
        p = os.path.join(preferred_dir, name)
        if os.path.exists(p):
            return p
    fallback_dir = str(PROJECT_ROOT / "student_resource" / "dataset" / "train")
    for name in names:
        p = os.path.join(fallback_dir, name)
        if os.path.exists(p):
            return p
    return os.path.join(preferred_dir, names[0])

S1_PATH = find_data_file(DATA_DIR, "source1.tsv", "train_source1.tsv")
S2_PATH = find_data_file(DATA_DIR, "source2.tsv", "train_source2.tsv")
S3_PATH = find_data_file(DATA_DIR, "source3.tsv", "train_source3.tsv")
GT_PATH = find_data_file(DATA_DIR, "ground_truth.tsv", "train_ground_truth.tsv")

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "entity_matcher_catboost.cbm"
)

TRAIN_PAIRS_PATH = os.path.join(
    OUTPUT_DIR,
    "training_pairs.parquet"
)

VAL_PRED_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_predictions.parquet"
)

FINAL_PATH = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_metrics.json"
)

# ------------------------------------------------------------
# IMPORTANT PARAMETERS
# ------------------------------------------------------------

# Number of S1 entities used for validation.
VAL_FRACTION = 0.10

# Positive : negative ratio.
NEGATIVE_RATIO = 4

# Number of candidates retained from each blocking rule
# for each S1 entity.
TOPK_PER_RULE = 30

# Final candidate limit per S1/source.
FINAL_TOPK = 100

# Batch size while generating pair features.
FEATURE_BATCH_SIZE = 100_000

# CatBoost
CATBOOST_ITERATIONS = 1200
CATBOOST_DEPTH = 8
CATBOOST_LEARNING_RATE = 0.05

# Classification threshold will be selected using validation.
THRESHOLDS = np.arange(
    0.20,
    0.96,
    0.02
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print(f"S1: {S1_PATH}")
print(f"S2: {S2_PATH}")
print(f"S3: {S3_PATH}")
print(f"GT: {GT_PATH}")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("S1:", s1.shape)
print("S2:", s2.shape)
print("S3:", s3.shape)
print("GT:", gt.shape)


# ============================================================
# 2. NORMALIZATION
# ============================================================

def normalize_unicode(text):
    """
    Unicode normalization.

    Important:
    We DO NOT transliterate Indian scripts to English.

    Marathi/Hindi -> Devanagari
    Telugu -> Telugu
    Kannada -> Kannada
    Tamil -> Tamil
    etc.

    Character similarity still works across noisy versions
    within the same script.
    """

    if text is None:
        return ""

    text = str(text)

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    # Normalize punctuation to spaces.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_name(text):
    return normalize_unicode(text)


def normalize_address(text):
    return normalize_unicode(text)


print("Creating normalized fields...")


for df in [s1, s2, s3]:

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_name)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_address)
    )


# ============================================================
# 3. SCRIPT DETECTION
# ============================================================

SCRIPT_PATTERNS = {
    "DEVANAGARI": r"[\u0900-\u097F]",
    "BENGALI": r"[\u0980-\u09FF]",
    "GURMUKHI": r"[\u0A00-\u0A7F]",
    "GUJARATI": r"[\u0A80-\u0AFF]",
    "ORIYA": r"[\u0B00-\u0B7F]",
    "TAMIL": r"[\u0B80-\u0BFF]",
    "TELUGU": r"[\u0C00-\u0C7F]",
    "KANNADA": r"[\u0C80-\u0CFF]",
    "MALAYALAM": r"[\u0D00-\u0D7F]",
}


def detect_script(text):

    if not text:
        return "EMPTY"

    counts = {}

    for script, pattern in SCRIPT_PATTERNS.items():

        count = len(
            re.findall(
                pattern,
                text
            )
        )

        if count > 0:
            counts[script] = count

    latin_count = len(
        re.findall(
            r"[a-zA-Z]",
            text
        )
    )

    if latin_count > 0:
        counts["LATIN"] = latin_count

    if not counts:
        return "OTHER"

    if len(counts) > 1:
        return "MIXED"

    return max(
        counts,
        key=counts.get
    )


for df in [s1, s2, s3]:

    df["name_script"] = (
        df["business_name"]
        .fillna("")
        .map(detect_script)
    )

    df["address_script"] = (
        df["business_address"]
        .fillna("")
        .map(detect_script)
    )


# ============================================================
# 4. NUMERIC TOKEN EXTRACTION
# ============================================================

def extract_numbers(text):

    if not text:
        return set()

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


def extract_postal(text, country):

    nums = extract_numbers(text)

    if not nums:
        return ""

    # US ZIP
    if country == "US":

        for n in nums:

            if len(n) == 5:
                return n

    # India PIN
    if country == "India":

        for n in nums:

            if len(n) == 6:
                return n

    return ""


for df in [s1, s2, s3]:

    df["number_tokens"] = (
        df["address_norm"]
        .map(extract_numbers)
    )

    df["postal"] = [
        extract_postal(
            addr,
            country
        )
        for addr, country in zip(
            df["address_norm"],
            df["country"]
        )
    ]


# ============================================================
# 5. TOKENIZATION
# ============================================================

def tokenize(text):

    if not text:
        return set()

    return set(
        x for x in text.split()
        if x
    )


for df in [s1, s2, s3]:

    df["name_tokens"] = (
        df["name_norm"]
        .map(tokenize)
    )

    df["address_tokens"] = (
        df["address_norm"]
        .map(tokenize)
    )


# ============================================================
# 6. GROUND TRUTH
# ============================================================

print("=" * 80)
print("PROCESSING GROUND TRUTH")
print("=" * 80)


def parse_gt_ids(x):

    if not x:
        return []

    return [
        y.strip()
        for y in str(x).split(",")
        if y.strip()
    ]


gt["matched_list"] = (
    gt["matched_entity_ids"]
    .map(parse_gt_ids)
)


# ------------------------------------------------------------
# Create lookup:
#
# S1 -> set(S2/S3 IDs)
# ------------------------------------------------------------

gt_map = dict(
    zip(
        gt["source1_entity_id"],
        gt["matched_list"]
    )
)


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
# ============================================================

all_s1_ids = s1["entity_id"].values

rng = np.random.default_rng(SEED)

shuffled = all_s1_ids.copy()

rng.shuffle(shuffled)

split_point = int(
    len(shuffled) *
    (1.0 - VAL_FRACTION)
)

train_s1_ids = set(
    shuffled[:split_point]
)

val_s1_ids = set(
    shuffled[split_point:]
)

print(
    "Training S1:",
    len(train_s1_ids)
)

print(
    "Validation S1:",
    len(val_s1_ids)
)


# ============================================================
# 8. BLOCKING INDEXES
# ============================================================

print("=" * 80)
print("BUILDING BLOCK INDEXES")
print("=" * 80)


def first_chars(text, n=6):

    if not text:
        return ""

    return text[:n]


def first_name_token(text):

    if not text:
        return ""

    parts = text.split()

    return parts[0] if parts else ""


def name_token_pair(text):

    if not text:
        return ""

    parts = sorted(set(text.split()))

    if len(parts) >= 2:
        return (
            parts[0] + "|" +
            parts[1]
        )

    if len(parts) == 1:
        return parts[0]

    return ""


def postal_name4(postal, name):

    if not postal:
        return ""

    return (
        postal +
        "|" +
        first_chars(name, 4)
    )


def build_blocks(df):

    blocks = {
        "postal_name4": defaultdict(list),
        "name_prefix6": defaultdict(list),
        "name_exact": defaultdict(list),
        "token_pair": defaultdict(list),
        "postal": defaultdict(list)
    }

    for idx, row in enumerate(
        df.itertuples(index=False)
    ):

        name = row.name_norm
        postal = row.postal

        k = postal_name4(
            postal,
            name
        )

        if k:
            blocks["postal_name4"][k].append(idx)

        k = first_chars(
            name,
            6
        )

        if k:
            blocks["name_prefix6"][k].append(idx)

        if name:
            blocks["name_exact"][name].append(idx)

        k = name_token_pair(name)

        if k:
            blocks["token_pair"][k].append(idx)

        if postal:
            blocks["postal"][postal].append(idx)

    return blocks


s2_blocks = build_blocks(s2)
s3_blocks = build_blocks(s3)

print("S2 indexes ready")
print("S3 indexes ready")


# Pre-convert target DataFrames to records for fast dictionary indexing
s1_records = s1.to_dict("records")
s2_records = s2.to_dict("records")
s3_records = s3.to_dict("records")


# ============================================================
# 9. CHEAP CANDIDATE SCORE
# ============================================================

def cheap_candidate_score(
    s1_row,
    target_row
):

    name_score = ratio(
        s1_row["name_norm"],
        target_row["name_norm"]
    ) / 100.0

    addr_score = ratio(
        s1_row["address_norm"],
        target_row["address_norm"]
    ) / 100.0

    token_name = token_set_ratio(
        s1_row["name_norm"],
        target_row["name_norm"]
    ) / 100.0

    token_addr = token_set_ratio(
        s1_row["address_norm"],
        target_row["address_norm"]
    ) / 100.0

    postal_match = (
        1.0
        if (
            s1_row["postal"]
            and
            s1_row["postal"]
            ==
            target_row["postal"]
        )
        else 0.0
    )

    # Stronger weight on name.
    score = (
        0.40 * name_score +
        0.20 * token_name +
        0.25 * addr_score +
        0.10 * token_addr +
        0.05 * postal_match
    )

    return score


# ============================================================
# 10. CANDIDATE GENERATION
# ============================================================

def get_block_candidates(
    s1_row,
    target_records,
    blocks,
    topk_per_rule=TOPK_PER_RULE,
    final_topk=FINAL_TOPK
):

    candidate_ids = set()

    name = s1_row["name_norm"]
    postal = s1_row["postal"]

    keys = {
        "postal_name4": postal_name4(
            postal,
            name
        ),

        "name_prefix6": first_chars(
            name,
            6
        ),

        "name_exact": name,

        "token_pair": name_token_pair(
            name
        ),

        "postal": postal
    }

    # --------------------------------------------------------
    # Collect candidates from each block.
    # --------------------------------------------------------

    for rule, key in keys.items():

        if not key:
            continue

        bucket = blocks[rule].get(
            key,
            []
        )

        if not bucket:
            continue

        # Do NOT dump huge buckets into candidate matrix.
        #
        # If bucket is small -> take all.
        # If huge -> random subset first.
        #
        # We later rank them with cheap fuzzy matching.

        if len(bucket) > topk_per_rule:

            step = max(
                1,
                len(bucket) //
                topk_per_rule
            )

            selected = bucket[
                ::step
            ][:topk_per_rule]

        else:

            selected = bucket

        candidate_ids.update(
            selected
        )

    if not candidate_ids:
        return []

    # --------------------------------------------------------
    # Rank candidates using cheap similarity.
    # --------------------------------------------------------

    candidates = []

    for idx in candidate_ids:

        target = target_records[idx]

        score = cheap_candidate_score(
            s1_row,
            target
        )

        candidates.append(
            (
                score,
                idx
            )
        )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return [
        idx
        for _, idx in
        candidates[:final_topk]
    ]


# ============================================================
# 11. PAIRWISE FEATURES
# ============================================================

FEATURE_COLUMNS = [

    # Exact
    "country_exact",
    "name_exact",
    "address_exact",
    "postal_exact",

    # String similarity
    "name_ratio",
    "name_wratio",
    "name_token_sort",
    "name_token_set",

    "address_ratio",
    "address_wratio",
    "address_token_sort",
    "address_token_set",

    # Token features
    "name_token_jaccard",
    "address_token_jaccard",

    # Numeric
    "number_overlap",

    # Length
    "name_length_diff",
    "address_length_diff",

    "name_length_ratio",
    "address_length_ratio",

    # Script
    "name_script_same",
    "address_script_same",

    # Structural
    "name_contains",
    "address_contains",

    # Composite
    "name_address_score"
]


def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return ratio(
        a,
        b
    ) / 100.0


def safe_wratio(a, b):

    if not a or not b:
        return 0.0

    return WRatio(
        a,
        b
    ) / 100.0


def safe_token_sort(a, b):

    if not a or not b:
        return 0.0

    return token_sort_ratio(
        a,
        b
    ) / 100.0


def safe_token_set(a, b):

    if not a or not b:
        return 0.0

    return token_set_ratio(
        a,
        b
    ) / 100.0


def jaccard(a, b):

    if not a or not b:
        return 0.0

    union = len(
        a | b
    )

    if union == 0:
        return 0.0

    return len(
        a & b
    ) / union


def overlap_ratio(a, b):

    if not a or not b:
        return 0.0

    return len(
        a & b
    ) / max(
        1,
        min(
            len(a),
            len(b)
        )
    )


def pair_features(
    a,
    b
):

    name_a = a["name_norm"]
    name_b = b["name_norm"]

    addr_a = a["address_norm"]
    addr_b = b["address_norm"]

    name_ratio_v = safe_ratio(
        name_a,
        name_b
    )

    addr_ratio_v = safe_ratio(
        addr_a,
        addr_b
    )

    name_token = safe_token_set(
        name_a,
        name_b
    )

    addr_token = safe_token_set(
        addr_a,
        addr_b
    )

    postal_exact = int(
        bool(
            a["postal"]
        )
        and
        a["postal"]
        ==
        b["postal"]
    )

    name_contains = int(
        bool(name_a)
        and
        (
            name_a in name_b
            or
            name_b in name_a
        )
    )

    address_contains = int(
        bool(addr_a)
        and
        (
            addr_a in addr_b
            or
            addr_b in addr_a
        )
    )

    name_len_a = len(name_a)
    name_len_b = len(name_b)

    addr_len_a = len(addr_a)
    addr_len_b = len(addr_b)

    features = {

        "country_exact": int(
            a["country"] ==
            b["country"]
        ),

        "name_exact": int(
            name_a ==
            name_b
            and
            bool(name_a)
        ),

        "address_exact": int(
            addr_a ==
            addr_b
            and
            bool(addr_a)
        ),

        "postal_exact": postal_exact,

        "name_ratio": name_ratio_v,

        "name_wratio": safe_wratio(
            name_a,
            name_b
        ),

        "name_token_sort": safe_token_sort(
            name_a,
            name_b
        ),

        "name_token_set": name_token,

        "address_ratio": addr_ratio_v,

        "address_wratio": safe_wratio(
            addr_a,
            addr_b
        ),

        "address_token_sort": safe_token_sort(
            addr_a,
            addr_b
        ),

        "address_token_set": addr_token,

        "name_token_jaccard": jaccard(
            a["name_tokens"],
            b["name_tokens"]
        ),

        "address_token_jaccard": jaccard(
            a["address_tokens"],
            b["address_tokens"]
        ),

        "number_overlap": overlap_ratio(
            a["number_tokens"],
            b["number_tokens"]
        ),

        "name_length_diff": abs(
            name_len_a -
            name_len_b
        ),

        "address_length_diff": abs(
            addr_len_a -
            addr_len_b
        ),

        "name_length_ratio":
            min(
                name_len_a,
                name_len_b
            ) /
            max(
                1,
                max(
                    name_len_a,
                    name_len_b
                )
            ),

        "address_length_ratio":
            min(
                addr_len_a,
                addr_len_b
            ) /
            max(
                1,
                max(
                    addr_len_a,
                    addr_len_b
                )
            ),

        "name_script_same": int(
            a["name_script"] ==
            b["name_script"]
        ),

        "address_script_same": int(
            a["address_script"] ==
            b["address_script"]
        ),

        "name_contains": name_contains,

        "address_contains": address_contains,

        "name_address_score":
            (
                0.55 *
                name_ratio_v
                +
                0.45 *
                addr_ratio_v
            )
    }

    return features


# ============================================================
# 12. CREATE POSITIVE TRAINING PAIRS
# ============================================================

print("=" * 80)
print("CREATING POSITIVE PAIRS")
print("=" * 80)


s1_index = {
    x: i
    for i, x in enumerate(
        s1["entity_id"]
    )
}


s2_index = {
    x: i
    for i, x in enumerate(
        s2["entity_id"]
    )
}


s3_index = {
    x: i
    for i, x in enumerate(
        s3["entity_id"]
    )
}


def create_positive_pairs():

    rows = []

    for s1_id, matches in gt_map.items():

        if s1_id not in train_s1_ids:
            continue

        if not matches:
            continue

        if s1_id not in s1_index:
            continue

        s1_idx = s1_index[s1_id]

        for target_id in matches:

            if target_id.startswith("S2-"):

                target_idx = s2_index.get(
                    target_id
                )

                if target_idx is None:
                    continue

                rows.append(
                    (
                        s1_idx,
                        target_idx,
                        "S2",
                        1
                    )
                )

            elif target_id.startswith("S3-"):

                target_idx = s3_index.get(
                    target_id
                )

                if target_idx is None:
                    continue

                rows.append(
                    (
                        s1_idx,
                        target_idx,
                        "S3",
                        1
                    )
                )

    return rows


positive_pairs = create_positive_pairs()

print(
    "Positive pairs:",
    len(positive_pairs)
)


# ============================================================
# 13. HARD NEGATIVE GENERATION
# ============================================================

def generate_training_pairs():

    rows = []

    positive_lookup = defaultdict(set)

    for (
        s1_idx,
        target_idx,
        source,
        label
    ) in positive_pairs:

        positive_lookup[
            (
                s1_idx,
                source
            )
        ].add(
            target_idx
        )

    # --------------------------------------------------------
    # Process positives.
    # For every positive, retrieve candidate pool and sample
    # hard negatives from the same blocks.
    # --------------------------------------------------------

    for counter, (
        s1_idx,
        target_idx,
        source,
        label
    ) in enumerate(
        positive_pairs
    ):

        if source == "S2":

            target_records = s2_records
            blocks = s2_blocks

        else:

            target_records = s3_records
            blocks = s3_blocks

        a = s1_records[s1_idx]

        candidate_indices = get_block_candidates(
            a,
            target_records,
            blocks
        )

        positive_set = positive_lookup[
            (
                s1_idx,
                source
            )
        ]

        negative_candidates = [
            x
            for x in candidate_indices
            if x not in positive_set
        ]

        # ----------------------------------------------------
        # Hard negatives:
        # candidates with high cheap similarity
        # ----------------------------------------------------

        negative_candidates = negative_candidates[
            :NEGATIVE_RATIO
        ]

        # Positive.
        positive_row = pair_features(
            a,
            target_records[target_idx]
        )

        positive_row["label"] = 1
        positive_row["source"] = source

        rows.append(
            positive_row
        )

        # Negatives.
        for neg_idx in negative_candidates:

            negative_row = pair_features(
                a,
                target_records[neg_idx]
            )

            negative_row["label"] = 0
            negative_row["source"] = source

            rows.append(
                negative_row
            )

        if counter % 10_000 == 0:

            print(
                "Processed positives:",
                counter,
                "/",
                len(positive_pairs)
            )

    return pd.DataFrame(rows)


train_df = generate_training_pairs()

print(
    "Training dataframe:",
    train_df.shape
)

print(
    train_df["label"].value_counts()
)


# ============================================================
# 14. REMOVE DUPLICATES
# ============================================================

train_df = train_df.drop_duplicates(
    subset=FEATURE_COLUMNS + ["label"]
)

train_df.reset_index(
    drop=True,
    inplace=True
)

print(
    "After deduplication:",
    train_df.shape
)


# ============================================================
# 15. SAVE TRAINING DATA
# ============================================================

train_df.to_parquet(
    TRAIN_PAIRS_PATH,
    index=False
)


# ============================================================
# 16. TRAIN / VALIDATION DATA
# ============================================================

X = train_df[
    FEATURE_COLUMNS
]

y = train_df[
    "label"
].astype(
    np.int8
)

# Stratified random split at PAIR level is not ideal,
# because entity leakage is possible.
#
# We already created positives only from train S1 entities.
#
# Therefore validation below is generated independently
# from validation S1 entities.


# ============================================================
# 17. CATBOOST TRAINING
# ============================================================

print("=" * 80)
print("TRAINING CATBOOST")
print("=" * 80)

# Check GPU availability and configure CatBoost
cb_kwargs = {
    "iterations": CATBOOST_ITERATIONS,
    "depth": CATBOOST_DEPTH,
    "learning_rate": CATBOOST_LEARNING_RATE,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": SEED,
    "task_type": "GPU",
    "devices": "0",
    "gpu_ram_part": 0.85,
    "verbose": 100,
    "allow_writing_files": False
}

try:
    model = CatBoostClassifier(**cb_kwargs)
    train_pool = Pool(
        X,
        label=y
    )
    model.fit(train_pool)
except Exception as e:
    print(f"\nGPU training failed or unavailable ({e}). Falling back to CPU...")
    cb_kwargs["task_type"] = "CPU"
    cb_kwargs.pop("devices", None)
    cb_kwargs.pop("gpu_ram_part", None)
    cb_kwargs["thread_count"] = -1
    model = CatBoostClassifier(**cb_kwargs)
    train_pool = Pool(
        X,
        label=y
    )
    model.fit(train_pool)

model.save_model(
    MODEL_PATH
)

print(
    "Model saved:",
    MODEL_PATH
)


# ============================================================
# 18. VALIDATION CANDIDATES
# ============================================================

print("=" * 80)
print("GENERATING VALIDATION CANDIDATES")
print("=" * 80)


def build_validation_candidates():

    rows = []

    val_ids = list(
        val_s1_ids
    )

    for counter, s1_id in enumerate(
        val_ids
    ):

        s1_idx = s1_index[
            s1_id
        ]

        a = s1_records[
            s1_idx
        ]

        gt_matches = set(
            gt_map.get(
                s1_id,
                []
            )
        )

        for source, target_records, blocks in [

            (
                "S2",
                s2_records,
                s2_blocks
            ),

            (
                "S3",
                s3_records,
                s3_blocks
            )

        ]:

            candidates = get_block_candidates(
                a,
                target_records,
                blocks
            )

            for target_idx in candidates:

                target = target_records[target_idx]
                target_id = target["entity_id"]

                features = pair_features(
                    a,
                    target
                )

                features[
                    "s1_id"
                ] = s1_id

                features[
                    "target_id"
                ] = target_id

                features[
                    "source"
                ] = source

                features[
                    "label"
                ] = int(
                    target_id
                    in gt_matches
                )

                rows.append(
                    features
                )

        if counter % 5000 == 0:

            print(
                "Validation entities:",
                counter,
                "/",
                len(val_ids)
            )

    return pd.DataFrame(
        rows
    )


val_df = build_validation_candidates()

print(
    "Validation candidate pairs:",
    val_df.shape
)


# ============================================================
# 19. VALIDATION PREDICTIONS
# ============================================================

val_X = val_df[
    FEATURE_COLUMNS
]

val_df["probability"] = model.predict_proba(
    val_X
)[:, 1]


val_df.to_parquet(
    VAL_PRED_PATH,
    index=False
)


# ============================================================
# 20. MACRO F0.5
# ============================================================

def fbeta(
    precision,
    recall,
    beta=0.5
):

    if precision <= 0 and recall <= 0:
        return 0.0

    beta2 = beta ** 2

    denominator = (
        beta2 * precision
        +
        recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta2)
        *
        precision
        *
        recall
        /
        denominator
    )


def macro_f05(
    prediction_df,
    threshold
):

    scores = []

    grouped = prediction_df.groupby(
        "s1_id"
    )

    for s1_id, group in grouped:

        predicted = set(
            group.loc[
                group["probability"]
                >= threshold,
                "target_id"
            ]
        )

        actual = set(
            gt_map.get(
                s1_id,
                []
            )
        )

        tp = len(
            predicted &
            actual
        )

        fp = len(
            predicted -
            actual
        )

        fn = len(
            actual -
            predicted
        )

        precision = (
            tp /
            (tp + fp)
            if
            (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp /
            (tp + fn)
            if
            (tp + fn) > 0
            else 1.0
        )

        score = fbeta(
            precision,
            recall,
            beta=0.5
        )

        scores.append(
            score
        )

    # --------------------------------------------------------
    # Important:
    # Add validation singletons that generated zero
    # candidates.
    # --------------------------------------------------------

    evaluated_ids = set(
        prediction_df[
            "s1_id"
        ]
    )

    missing_ids = (
        set(val_s1_ids)
        -
        evaluated_ids
    )

    for s1_id in missing_ids:

        actual = set(
            gt_map.get(
                s1_id,
                []
            )
        )

        if len(actual) == 0:
            scores.append(1.0)
        else:
            scores.append(0.0)

    return float(
        np.mean(scores)
    )


# ============================================================
# 21. THRESHOLD SEARCH
# ============================================================

print("=" * 80)
print("THRESHOLD SEARCH")
print("=" * 80)

threshold_results = []

for threshold in THRESHOLDS:

    score = macro_f05(
        val_df,
        threshold
    )

    threshold_results.append(
        {
            "threshold": float(
                threshold
            ),
            "macro_f05": score
        }
    )

threshold_results = pd.DataFrame(
    threshold_results
)

best_row = threshold_results.loc[
    threshold_results["macro_f05"].idxmax()
]

BEST_THRESHOLD = float(
    best_row["threshold"]
)

BEST_SCORE = float(
    best_row["macro_f05"]
)

print(
    threshold_results
)

print(
    "\nBEST THRESHOLD:",
    BEST_THRESHOLD
)

print(
    "BEST VALIDATION MACRO F0.5:",
    BEST_SCORE
)


# ============================================================
# 22. PRECISION / RECALL AT BEST THRESHOLD
# ============================================================

def pair_metrics(
    df,
    threshold
):

    pred = (
        df["probability"]
        >= threshold
    )

    actual = (
        df["label"]
        == 1
    )

    tp = int(
        (pred & actual).sum()
    )

    fp = int(
        (pred & ~actual).sum()
    )

    fn = int(
        (~pred & actual).sum()
    )

    precision = (
        tp /
        (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp /
        (tp + fn)
        if tp + fn
        else 0
    )

    f05 = fbeta(
        precision,
        recall,
        0.5
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f05": f05
    }


metrics = pair_metrics(
    val_df,
    BEST_THRESHOLD
)

print(
    json.dumps(
        metrics,
        indent=4
    )
)


# ============================================================
# 23. SAVE METRICS
# ============================================================

final_metrics = {

    "best_threshold":
        BEST_THRESHOLD,

    "validation_macro_f05":
        BEST_SCORE,

    "pair_precision":
        metrics["precision"],

    "pair_recall":
        metrics["recall"],

    "pair_f05":
        metrics["f05"],

    "validation_entities":
        len(val_s1_ids),

    "validation_candidates":
        len(val_df)
}

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=4
    )


# ============================================================
# 24. TEST / FINAL INFERENCE
# ============================================================

print("=" * 80)
print("FINAL TEST INFERENCE")
print("=" * 80)


def generate_test_predictions():

    result = defaultdict(list)

    all_ids = s1[
        "entity_id"
    ].tolist()

    for counter, s1_id in enumerate(
        all_ids
    ):

        s1_idx = s1_index[
            s1_id
        ]

        a = s1_records[
            s1_idx
        ]

        # ----------------------------------------------------
        # S2 & S3
        # ----------------------------------------------------

        for source, target_records, blocks in [

            (
                "S2",
                s2_records,
                s2_blocks
            ),

            (
                "S3",
                s3_records,
                s3_blocks
            )

        ]:

            candidate_indices = get_block_candidates(
                a,
                target_records,
                blocks
            )

            if not candidate_indices:
                continue

            feature_rows = []
            metadata = []

            for target_idx in candidate_indices:

                target = target_records[target_idx]

                feature_rows.append(
                    pair_features(
                        a,
                        target
                    )
                )

                metadata.append(
                    (
                        source,
                        target["entity_id"]
                    )
                )

            feature_df = pd.DataFrame(
                feature_rows
            )

            probabilities = model.predict_proba(
                feature_df[
                    FEATURE_COLUMNS
                ]
            )[:, 1]

            for (
                (source_name, target_id),
                probability
            ) in zip(
                metadata,
                probabilities
            ):

                if probability >= BEST_THRESHOLD:

                    result[s1_id].append(
                        (
                            probability,
                            target_id
                        )
                    )

        if counter % 10_000 == 0:

            print(
                "Processed:",
                counter,
                "/",
                len(all_ids)
            )

    return result


prediction_map = generate_test_predictions()


# ============================================================
# 25. CREATE SUBMISSION
# ============================================================

print("=" * 80)
print("CREATING SUBMISSION")
print("=" * 80)


submission_rows = []

for s1_id in s1[
    "entity_id"
]:

    predictions = prediction_map.get(
        s1_id,
        []
    )

    # --------------------------------------------------------
    # Sort highest probability first.
    # --------------------------------------------------------

    predictions.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    matched_ids = [
        target_id
        for _, target_id
        in predictions
    ]

    submission_rows.append(
        {
            "source1_entity_id":
                s1_id,

            "matched_entity_ids":
                ",".join(
                    matched_ids
                )
        }
    )


submission = pd.DataFrame(
    submission_rows
)


# ------------------------------------------------------------
# Ensure exactly one row per S1.
# ------------------------------------------------------------

assert len(
    submission
) == len(s1)

assert submission[
    "source1_entity_id"
].is_unique


# ------------------------------------------------------------
# Write TSV.
# ------------------------------------------------------------

submission.to_csv(
    FINAL_PATH,
    sep="\t",
    index=False
)


print(
    "FINAL SUBMISSION:",
    FINAL_PATH
)

print(
    submission.head()
)


# ============================================================
# 26. SUMMARY
# ============================================================

print("=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Validation score:",
    BEST_SCORE
)

print(
    "Threshold:",
    BEST_THRESHOLD
)

print(
    "Submission:",
    FINAL_PATH
)
