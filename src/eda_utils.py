"""
eda_utils.py - Modular utility functions for EDA and Entity Resolution diagnostics
Amazon ML Challenge
"""

import os
import re
import gc
import json
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from rapidfuzz import fuzz
from unidecode import unidecode


# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

EXPECTED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]

GT_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids"
]

ABBREVIATIONS = {
    # Corporate
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "co": "company",
    "company": "company",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "llc": "limited liability company",

    # Address
    "rd": "road",
    "road": "road",
    "st": "street",
    "street": "street",
    "ave": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "ln": "lane",
    "lane": "lane",
    "dr": "drive",
    "drive": "drive",
    "hwy": "highway",
    "highway": "highway",
    "apt": "apartment",
    "apartment": "apartment",
    "fl": "floor",
    "floor": "floor",
    "ste": "suite",
    "suite": "suite",
}

SCRIPT_RANGES = {
    "DEVANAGARI": ("\u0900", "\u097F"),
    "BENGALI": ("\u0980", "\u09FF"),
    "GURMUKHI": ("\u0A00", "\u0A7F"),
    "GUJARATI": ("\u0A80", "\u0AFF"),
    "ORIYA": ("\u0B00", "\u0B7F"),
    "TAMIL": ("\u0B80", "\u0BFF"),
    "TELUGU": ("\u0C00", "\u0C7F"),
    "KANNADA": ("\u0C80", "\u0CFF"),
    "MALAYALAM": ("\u0D00", "\u0D7F"),
    "ARABIC": ("\u0600", "\u06FF"),
    "CYRILLIC": ("\u0400", "\u04FF"),
}

NUMBER_PATTERN = re.compile(r"\d+(?:[-/]\d+)*")


# ============================================================
# BASIC READERS & SCHEMA VALIDATION
# ============================================================

def read_source_sample(path, nrows=10_000):
    """
    Read a sample for interactive EDA.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        nrows=nrows,
        keep_default_na=True
    )


def read_ground_truth_sample(path, nrows=10_000):
    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        nrows=nrows,
        keep_default_na=True
    )


def validate_schema(path, expected_columns, dataset_name):
    sample = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        nrows=5
    )

    actual = list(sample.columns)

    print(f"\n{'=' * 70}")
    print(dataset_name)
    print(f"{'=' * 70}")

    print("Columns:")
    print(actual)

    missing = set(expected_columns) - set(actual)
    extra = set(actual) - set(expected_columns)

    if missing:
        print("Missing columns:", missing)
    else:
        print("Missing columns: None")

    if extra:
        print("Extra columns:", extra)
    else:
        print("Extra columns: None")

    return sample


def count_rows_tsv(path):
    """
    Count data rows in a TSV without loading the entire dataframe.
    Assumes one header row.
    """
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


# ============================================================
# CHUNKED BASIC EDA & SUMMARY
# ============================================================

def basic_eda_chunked(path, dataset_name, chunk_size=250_000):
    total_rows = 0

    null_counts = Counter()
    unique_ids = set()

    name_counter = Counter()
    address_counter = Counter()
    country_counter = Counter()

    duplicate_name_rows = 0
    duplicate_address_rows = 0

    name_lengths = []
    address_lengths = []

    for chunk in tqdm(
        pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            chunksize=chunk_size,
            keep_default_na=True
        ),
        desc=f"EDA {dataset_name}"
    ):
        total_rows += len(chunk)

        # Nulls
        null_counts.update(
            chunk[EXPECTED_COLUMNS].isna().sum().to_dict()
        )

        # IDs
        unique_ids.update(chunk["entity_id"].dropna().tolist())

        # Frequencies
        name_counter.update(
            chunk["business_name"].dropna().value_counts().to_dict()
        )

        address_counter.update(
            chunk["business_address"].dropna().value_counts().to_dict()
        )

        country_counter.update(
            chunk["country"].dropna().value_counts().to_dict()
        )

        # Lengths
        name_lengths.extend(
            chunk["business_name"]
            .dropna()
            .str.len()
            .tolist()
        )

        address_lengths.extend(
            chunk["business_address"]
            .dropna()
            .str.len()
            .tolist()
        )

    result = {
        "dataset": dataset_name,
        "rows": total_rows,
        "unique_entity_id": len(unique_ids),

        "unique_business_name": len(name_counter),
        "unique_business_address": len(address_counter),

        "nulls": dict(null_counts),

        "duplicate_name_rows": sum(
            count - 1
            for count in name_counter.values()
            if count > 1
        ),

        "duplicate_address_rows": sum(
            count - 1
            for count in address_counter.values()
            if count > 1
        ),

        "top_names": name_counter.most_common(20),
        "top_addresses": address_counter.most_common(20),
        "countries": country_counter.most_common(),

        "name_length_stats": {
            "min": float(np.min(name_lengths)) if name_lengths else 0,
            "max": float(np.max(name_lengths)) if name_lengths else 0,
            "mean": float(np.mean(name_lengths)) if name_lengths else 0,
            "median": float(np.median(name_lengths)) if name_lengths else 0,
            "p95": float(np.percentile(name_lengths, 95)) if name_lengths else 0,
            "p99": float(np.percentile(name_lengths, 99)) if name_lengths else 0,
        },

        "address_length_stats": {
            "min": float(np.min(address_lengths)) if address_lengths else 0,
            "max": float(np.max(address_lengths)) if address_lengths else 0,
            "mean": float(np.mean(address_lengths)) if address_lengths else 0,
            "median": float(np.median(address_lengths)) if address_lengths else 0,
            "p95": float(np.percentile(address_lengths, 95)) if address_lengths else 0,
            "p99": float(np.percentile(address_lengths, 99)) if address_lengths else 0,
        },

        "name_counter": name_counter,
        "address_counter": address_counter,
    }

    return result


def print_eda_summary(stats):
    print("\n" + "=" * 80)
    print(stats["dataset"])
    print("=" * 80)

    print(f"Rows                  : {stats['rows']:,}")
    print(f"Unique entity IDs     : {stats['unique_entity_id']:,}")
    print(f"Unique business names : {stats['unique_business_name']:,}")
    print(f"Unique addresses      : {stats['unique_business_address']:,}")

    print("\nNulls:")
    for col, count in stats["nulls"].items():
        print(f"  {col:<20}: {count:,}")

    print("\nCountries:")
    for country, count in stats["countries"]:
        print(f"  {str(country):<20}: {count:,}")

    print("\nDuplicate rows caused by repeated names:")
    print(f"  {stats['duplicate_name_rows']:,}")

    print("\nDuplicate rows caused by repeated addresses:")
    print(f"  {stats['duplicate_address_rows']:,}")

    print("\nName length:")
    for k, v in stats["name_length_stats"].items():
        print(f"  {k:<10}: {v:.2f}")

    print("\nAddress length:")
    for k, v in stats["address_length_stats"].items():
        print(f"  {k:<10}: {v:.2f}")

    print("\nTop 20 business names:")
    for name, count in stats["top_names"]:
        print(f"  {count:>8,} | {name}")

    print("\nTop 20 addresses:")
    for address, count in stats["top_addresses"]:
        print(f"  {count:>8,} | {address}")


def null_percentage(stats):
    rows = stats["rows"]
    records = []
    for col, count in stats["nulls"].items():
        records.append({
            "column": col,
            "null_count": count,
            "null_percent": 100 * count / rows if rows > 0 else 0
        })
    return pd.DataFrame(records)


def top_duplicate_names(stats, top_n=50):
    rows = [
        {
            "business_name": name,
            "frequency": count
        }
        for name, count in stats["name_counter"].most_common()
        if count > 1
    ]
    return pd.DataFrame(rows).head(top_n)


def top_duplicate_addresses(stats, top_n=50):
    rows = [
        {
            "business_address": address,
            "frequency": count
        }
        for address, count in stats["address_counter"].most_common()
        if count > 1
    ]
    return pd.DataFrame(rows).head(top_n)


# ============================================================
# TEXT NORMALIZATION & PREPROCESSING
# ============================================================

def normalize_unicode(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Unicode compatibility normalization
    text = unicodedata.normalize("NFKC", text)
    # Case normalization
    text = text.casefold()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text):
    text = normalize_unicode(text)
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    text = normalize_text(text)
    if not text:
        return []
    return text.split()


def expand_abbreviations(text):
    tokens = tokenize(text)
    expanded = [ABBREVIATIONS.get(token, token) for token in tokens]
    return " ".join(expanded)


def transliterate_text(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    return unidecode(str(text))


def merged_text(name, address):
    name = "" if pd.isna(name) else str(name)
    address = "" if pd.isna(address) else str(address)
    return f"{name} {address}".strip()


def add_text_features(df):
    df = df.copy()

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_text)
    )

    df["merged_norm"] = (
        df["name_norm"] + " " + df["address_norm"]
    ).str.strip()

    df["name_expanded"] = (
        df["business_name"]
        .fillna("")
        .map(expand_abbreviations)
    )

    df["address_expanded"] = (
        df["business_address"]
        .fillna("")
        .map(expand_abbreviations)
    )

    df["merged_expanded"] = (
        df["name_expanded"]
        + " "
        + df["address_expanded"]
    ).str.strip()

    df["name_translit"] = (
        df["business_name"]
        .fillna("")
        .map(transliterate_text)
        .map(normalize_text)
    )

    df["address_translit"] = (
        df["business_address"]
        .fillna("")
        .map(transliterate_text)
        .map(normalize_text)
    )

    df["merged_translit"] = (
        df["name_translit"]
        + " "
        + df["address_translit"]
    ).str.strip()

    return df


def token_preview(df, text_column="merged_norm", max_tokens=15):
    temp = df[text_column].fillna("").map(tokenize)
    result = pd.DataFrame(
        temp.map(
            lambda x: x[:max_tokens] + [""] * max(0, max_tokens - len(x))
        ).tolist(),
        columns=[f"token_{i+1}" for i in range(max_tokens)]
    )
    return result


def extract_numbers(text):
    if pd.isna(text):
        return []
    return NUMBER_PATTERN.findall(str(text))


def number_features(text):
    nums = extract_numbers(text)
    return {
        "numbers": nums,
        "number_count": len(nums)
    }


# ============================================================
# SCRIPT DETECTION
# ============================================================

def char_script(ch):
    code = ord(ch)
    for script, (start, end) in SCRIPT_RANGES.items():
        if ord(start) <= code <= ord(end):
            return script
    if ch.isascii() and ch.isalpha():
        return "LATIN"
    if ch.isdigit():
        return "DIGIT"
    return "OTHER"


def detect_script(text):
    if pd.isna(text):
        return "EMPTY"

    scripts = set()
    for ch in str(text):
        script = char_script(ch)
        if script not in {"DIGIT", "OTHER"} and not ch.isspace():
            scripts.add(script)

    if not scripts:
        return "EMPTY"
    if len(scripts) == 1:
        return next(iter(scripts))
    return "MIXED"


def script_distribution(path, dataset_name, chunk_size=250_000):
    counter = Counter()

    for chunk in tqdm(
        pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            usecols=[
                "business_name",
                "business_address"
            ],
            chunksize=chunk_size
        ),
        desc=f"Script analysis {dataset_name}"
    ):
        names = chunk["business_name"].fillna("")
        addresses = chunk["business_address"].fillna("")

        for value in names:
            counter[detect_script(value)] += 1

        for value in addresses:
            counter[detect_script(value)] += 1

    result = pd.DataFrame(
        counter.items(),
        columns=["script", "count"]
    ).sort_values(
        "count",
        ascending=False
    )

    result["percent"] = (
        result["count"] / result["count"].sum() * 100
    )

    return result


def script_pair_analysis(path, dataset_name, chunk_size=250_000):
    counter = Counter()

    for chunk in tqdm(
        pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            usecols=[
                "business_name",
                "business_address"
            ],
            chunksize=chunk_size
        ),
        desc=f"Script pair {dataset_name}"
    ):
        for name, address in zip(
            chunk["business_name"].fillna(""),
            chunk["business_address"].fillna("")
        ):
            key = (
                detect_script(name),
                detect_script(address)
            )
            counter[key] += 1

    result = pd.DataFrame(
        [
            {
                "name_script": k[0],
                "address_script": k[1],
                "count": v
            }
            for k, v in counter.items()
        ]
    ).sort_values(
        "count",
        ascending=False
    )

    result["percent"] = (
        result["count"] / result["count"].sum() * 100
    )

    return result


def find_non_latin_examples(path, n=100, chunk_size=250_000):
    rows = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        usecols=EXPECTED_COLUMNS,
        chunksize=chunk_size
    ):
        mask = (
            chunk["business_name"]
            .fillna("")
            .map(detect_script)
            .isin([
                "DEVANAGARI",
                "TELUGU",
                "KANNADA",
                "TAMIL",
                "MALAYALAM",
                "BENGALI",
                "GUJARATI",
                "GURMUKHI",
                "ORIYA",
                "MIXED"
            ])
        )

        found = chunk.loc[mask]
        if len(found):
            rows.append(found)

        if sum(len(x) for x in rows) >= n:
            break

    if not rows:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    return pd.concat(rows, ignore_index=True).head(n)


def find_transliteration_examples(path, n=100, chunk_size=250_000):
    records = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        usecols=EXPECTED_COLUMNS,
        chunksize=chunk_size
    ):
        for _, row in chunk.iterrows():
            name = row["business_name"]
            address = row["business_address"]

            name_script = detect_script(name)
            address_script = detect_script(address)

            if (
                name_script not in {"LATIN", "EMPTY"}
                or address_script not in {"LATIN", "EMPTY"}
            ):
                records.append({
                    "entity_id": row["entity_id"],
                    "name": name,
                    "address": address,
                    "name_script": name_script,
                    "address_script": address_script,
                    "name_translit": transliterate_text(name),
                    "address_translit": transliterate_text(address),
                })

            if len(records) >= n:
                return pd.DataFrame(records)

    return pd.DataFrame(records)


# ============================================================
# TOKEN & ABBREVIATION AGGREGATION
# ============================================================

def token_statistics(
    path,
    dataset_name,
    text_columns=("business_name", "business_address"),
    chunk_size=250_000,
    top_n=100
):
    counter = Counter()

    for chunk in tqdm(
        pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            usecols=list(text_columns),
            chunksize=chunk_size
        ),
        desc=f"Token analysis {dataset_name}"
    ):
        for col in text_columns:
            for value in chunk[col].fillna(""):
                tokens = tokenize(value)
                counter.update(tokens)

    top_tokens = pd.DataFrame(
        counter.most_common(top_n),
        columns=["token", "count"]
    )

    return top_tokens


def abbreviation_frequency(
    path,
    dataset_name,
    chunk_size=250_000
):
    counter = Counter()
    abbreviation_tokens = set(ABBREVIATIONS.keys())

    for chunk in tqdm(
        pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            usecols=[
                "business_name",
                "business_address"
            ],
            chunksize=chunk_size
        ),
        desc=f"Abbreviation analysis {dataset_name}"
    ):
        for col in ["business_name", "business_address"]:
            for value in chunk[col].fillna(""):
                tokens = tokenize(value)
                for token in tokens:
                    if token in abbreviation_tokens:
                        counter[token] += 1

    return pd.DataFrame(
        counter.most_common(),
        columns=["abbreviation", "count"]
    )


def text_length_sample(
    path,
    chunk_size=250_000,
    sample_per_chunk=5_000
):
    samples_name = []
    samples_address = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        usecols=[
            "business_name",
            "business_address"
        ],
        chunksize=chunk_size
    ):
        sample = chunk.sample(
            min(sample_per_chunk, len(chunk)),
            random_state=42
        )

        samples_name.extend(
            sample["business_name"]
            .fillna("")
            .str.len()
            .tolist()
        )

        samples_address.extend(
            sample["business_address"]
            .fillna("")
            .str.len()
            .tolist()
        )

    return (
        np.array(samples_name),
        np.array(samples_address)
    )


# ============================================================
# GROUND TRUTH ANALYSIS HELPERS
# ============================================================

def classify_match_pattern(row):
    s2 = row["s2_match_count"]
    s3 = row["s3_match_count"]

    if s2 == 0 and s3 == 0:
        return "NONE"
    elif s2 > 0 and s3 == 0:
        return "S2_ONLY"
    elif s2 == 0 and s3 > 0:
        return "S3_ONLY"
    else:
        return "S2_AND_S3"


def has_duplicate_match_ids(match_string):
    if pd.isna(match_string) or not str(match_string).strip():
        return False
    ids = str(match_string).split(",")
    return len(ids) != len(set(ids))


def validate_match_id_prefixes(match_string):
    if pd.isna(match_string) or not str(match_string).strip():
        return True
    ids = str(match_string).split(",")
    return all(
        x.startswith(("S2-", "S3-"))
        for x in ids
    )


def read_ids(path, id_column="entity_id", chunk_size=500_000):
    result = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype={id_column: "string"},
        usecols=[id_column],
        chunksize=chunk_size
    ):
        result.append(chunk[id_column])

    return pd.concat(result, ignore_index=True)


# ============================================================
# BLOCKING DIAGNOSTICS HELPERS
# ============================================================

def normalize_for_blocking(text):
    text = normalize_text(text)
    # Keep alphanumeric characters
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def first_token(text):
    tokens = tokenize(text)
    return tokens[0] if tokens else ""


def first_n_chars(text, n=4):
    text = re.sub(
        r"[^a-z0-9]",
        "",
        normalize_for_blocking(text)
    )
    return text[:n]


def extract_postal_code(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # India PIN
    m = re.search(r"\b\d{6}\b", text)
    if m:
        return m.group()
    # US ZIP / ZIP+4
    m = re.search(r"\b\d{5}(?:-\d{4})?\b", text)
    if m:
        return m.group()
    return ""


def add_blocking_features(df):
    df = df.copy()

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_for_blocking)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_for_blocking)
    )

    df["name_prefix4"] = (
        df["name_norm"]
        .map(lambda x: first_n_chars(x, 4))
    )

    df["name_prefix6"] = (
        df["name_norm"]
        .map(lambda x: first_n_chars(x, 6))
    )

    df["name_first_token"] = (
        df["name_norm"]
        .map(first_token)
    )

    df["postal_code"] = (
        df["business_address"]
        .fillna("")
        .map(extract_postal_code)
    )

    df["country_name_prefix"] = (
        df["country"].fillna("")
        + "_"
        + df["name_prefix4"]
    )

    df["country_postal"] = (
        df["country"].fillna("")
        + "_"
        + df["postal_code"]
    )

    return df


def block_size_analysis(
    path,
    key_function,
    key_name,
    chunk_size=250_000
):
    counter = Counter()

    for chunk in tqdm(
        pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            usecols=[
                "business_name",
                "business_address",
                "country"
            ],
            chunksize=chunk_size
        ),
        desc=f"Block analysis: {key_name}"
    ):
        temp = add_blocking_features(chunk)
        keys = key_function(temp)
        counter.update(keys.dropna().tolist())

    sizes = pd.Series(counter, name="block_size")
    return sizes


def estimate_pair_count_from_blocks(s1_keys, s2_keys):
    s1_counts = s1_keys.value_counts()
    s2_counts = s2_keys.value_counts()

    common = s1_counts.index.intersection(s2_counts.index)
    estimated_pairs = (
        s1_counts.loc[common] * s2_counts.loc[common]
    ).sum()

    return int(estimated_pairs)


def build_block_index(df, key_column):
    index = defaultdict(list)
    for row in df.itertuples(index=False):
        key = getattr(row, key_column)
        entity_id = row.entity_id
        if not pd.isna(key) and str(key).strip():
            index[key].append(entity_id)
    return index


def evaluate_blocking_on_sample(
    s1_df,
    s2_df,
    gt_df,
    s1_key,
    s2_key
):
    s1 = add_blocking_features(s1_df)
    s2 = add_blocking_features(s2_df)

    index = build_block_index(s2, s2_key)

    recovered = 0
    total_true = 0
    rows = []

    gt_map = dict(
        zip(
            gt_df["source1_entity_id"],
            gt_df["matched_entity_ids"]
        )
    )

    for row in s1.itertuples(index=False):
        s1_id = row.entity_id
        true_string = gt_map.get(s1_id)

        if pd.isna(true_string) or not str(true_string).strip():
            continue

        true_ids = set(str(true_string).split(","))
        key = getattr(row, s1_key)
        candidates = set(index.get(key, [])) if not pd.isna(key) else set()

        hit_count = len(true_ids.intersection(candidates))
        recovered += hit_count
        total_true += len(true_ids)

        rows.append({
            "entity_id": s1_id,
            "true_matches": len(true_ids),
            "recovered": hit_count,
            "block_candidates": len(candidates)
        })

    detail = pd.DataFrame(rows)
    recall = (recovered / total_true if total_true > 0 else 0)
    return recall, detail


def similarity_demo(name_a, name_b):
    return {
        "ratio": fuzz.ratio(name_a, name_b),
        "partial_ratio": fuzz.partial_ratio(name_a, name_b),
        "token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b),
        "token_set_ratio": fuzz.token_set_ratio(name_a, name_b),
    }
