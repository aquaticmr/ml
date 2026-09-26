"""
blocking_v2_utils.py - Modular normalization, feature engineering,
multi-pass blocking keys, inverted indexing, candidate pair generation,
and recall evaluation for Amazon ML Challenge Stage 2 (Blocking V2).
"""

import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from unidecode import unidecode
import unicodedata

# ------------------------------------------------------------
# 1. Unicode normalization & transliteration
# ------------------------------------------------------------

def normalize_unicode(x):
    if pd.isna(x):
        return ""

    x = str(x)
    x = unicodedata.normalize("NFKC", x)
    x = unidecode(x)
    x = x.lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# ------------------------------------------------------------
# 2. Abbreviation normalization
# ------------------------------------------------------------

ABBR = {
    "pvt": "private",
    "pvtltd": "private limited",
    "ltd": "limited",
    "llc": "limited liability company",
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "corporation": "corporation",

    "rd": "road",
    "st": "street",
    "dr": "drive",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "hwy": "highway",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",

    "no": "number",
    "nr": "near",
    "opp": "opposite",

    "mt": "mount",
    "ctr": "center",
    "pl": "place",
    "ct": "court",
    "sq": "square",
}


def expand_abbreviations(text):
    if not text:
        return ""
    tokens = text.split()
    return " ".join(ABBR.get(token, token) for token in tokens)


# ------------------------------------------------------------
# 3. Name normalization & core extraction
# ------------------------------------------------------------

def normalize_name(x):
    x = normalize_unicode(x)
    x = expand_abbreviations(x)
    return x


def normalize_name_core(x):
    x = normalize_name(x)
    tokens = x.split()
    corporate = {
        "private",
        "limited",
        "company",
        "corporation",
        "incorporated",
        "llc",
        "liability",
    }
    filtered = [t for t in tokens if t not in corporate]
    return " ".join(filtered) if filtered else x


# ------------------------------------------------------------
# 4. Address normalization
# ------------------------------------------------------------

def normalize_address(x):
    x = normalize_unicode(x)
    x = expand_abbreviations(x)
    return x


# ------------------------------------------------------------
# 5. Numeric component
# ------------------------------------------------------------

def extract_numbers(x):
    if not x:
        return ""
    nums = re.findall(r"\d+", str(x))
    return " ".join(nums)


# ------------------------------------------------------------
# 6. Postal code
# ------------------------------------------------------------

def extract_postal(x, country):
    if not x:
        return ""
    country_str = "" if pd.isna(country) else str(country).strip().upper()

    # India 6-digit PIN
    if country_str in ("INDIA", "IN"):
        m = re.findall(r"\b[1-9][0-9]{5}\b", str(x))
        if m:
            return m[-1]

    # US 5-digit ZIP
    if country_str in ("US", "USA", "UNITED STATES"):
        m = re.findall(r"\b\d{5}(?:-\d{4})?\b", str(x))
        if m:
            return m[-1][:5]

    return ""


# ------------------------------------------------------------
# 7. Geography tokens
# ------------------------------------------------------------

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy",
}


# ------------------------------------------------------------
# 8. Feature engineering on DataFrame
# ------------------------------------------------------------

def engineer_features(df):
    df = df.copy()

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_name)
    )

    df["name_core"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_name_core)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_address)
    )

    df["numbers"] = (
        df["address_norm"]
        .map(extract_numbers)
    )

    df["postal"] = [
        extract_postal(a, c)
        for a, c in zip(df["address_norm"], df["country"])
    ]

    # First / last meaningful name tokens
    df["name_tokens"] = df["name_core"].str.split()
    df["name_first"] = df["name_tokens"].str[0].fillna("")
    df["name_last"] = df["name_tokens"].str[-1].fillna("")

    df["name_chars"] = df["name_core"].str.replace(" ", "", regex=False)
    df["name_prefix4"] = df["name_chars"].str[:4]
    df["name_prefix6"] = df["name_chars"].str[:6]

    return df


# ------------------------------------------------------------
# 9. Block keys generation
# ------------------------------------------------------------

def create_block_keys(df):
    d = {}

    country = df["country"].fillna("").astype(str).str.strip().str.upper()

    d["name_exact"] = country + "|" + df["name_norm"].astype(str)
    d["name_core"] = country + "|" + df["name_core"].astype(str)

    d["postal_name4"] = (
        country + "|" + df["postal"].astype(str) + "|" + df["name_prefix4"].astype(str)
    )
    d["postal_name6"] = (
        country + "|" + df["postal"].astype(str) + "|" + df["name_prefix6"].astype(str)
    )
    d["postal_first"] = (
        country + "|" + df["postal"].astype(str) + "|" + df["name_first"].astype(str)
    )

    d["first_last"] = (
        country + "|" + df["name_first"].astype(str) + "|" + df["name_last"].astype(str)
    )
    d["number_first"] = (
        country + "|" + df["numbers"].astype(str) + "|" + df["name_first"].astype(str)
    )
    d["number_name4"] = (
        country + "|" + df["numbers"].astype(str) + "|" + df["name_prefix4"].astype(str)
    )

    block_df = pd.DataFrame(d)

    # Clean out empty/uninformative keys
    for col in block_df.columns:
        block_df.loc[block_df[col].str.endswith("|"), col] = ""
        block_df.loc[block_df[col].isin(["", "|", "||"]), col] = ""

    return block_df


# ------------------------------------------------------------
# 10. Block quality statistics
# ------------------------------------------------------------

def block_statistics(block_df):
    rows = []
    for col in block_df.columns:
        vc = block_df[col].value_counts()
        vc = vc[vc.index != ""]

        if len(vc) == 0:
            rows.append({
                "rule": col,
                "blocks": 0,
                "mean": 0,
                "median": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0,
                "max": 0,
                "blocks_gt_100": 0,
                "blocks_gt_1000": 0,
            })
            continue

        rows.append({
            "rule": col,
            "blocks": len(vc),
            "mean": float(vc.mean()),
            "median": float(vc.median()),
            "p90": float(vc.quantile(0.90)),
            "p95": float(vc.quantile(0.95)),
            "p99": float(vc.quantile(0.99)),
            "max": int(vc.max()),
            "blocks_gt_100": int((vc > 100).sum()),
            "blocks_gt_1000": int((vc > 1000).sum()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 11. Inverted index builder with size capping
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

    groups = (
        temp.groupby("key", sort=False)["idx"]
        .apply(np.asarray)
        .to_dict()
    )
    return groups


# ------------------------------------------------------------
# 12. Candidate pair generator (chunked, memory-safe)
# ------------------------------------------------------------

def generate_candidates(
    source_df,
    target_df,
    source_blocks,
    target_blocks,
    rules,
    max_block_size=100,
    chunk_size=100_000
):
    indexes = {}
    for rule in rules:
        indexes[rule] = build_inverted_index(
            target_blocks[rule],
            max_block_size=max_block_size
        )

    results = []
    source_n = len(source_df)

    for start in range(0, source_n, chunk_size):
        end = min(start + chunk_size, source_n)
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

        if start % (chunk_size * 5) == 0 or end == source_n:
            print(f"Processed {end:,}/{source_n:,} source rows...")

    if not results:
        return np.empty((0, 2), dtype=np.int64)

    return np.vstack(results)


# ------------------------------------------------------------
# 13. Candidate recall calculation
# ------------------------------------------------------------

def calculate_candidate_recall(candidates, ground_truth):
    if len(candidates) == 0:
        return {
            "candidate_pairs": 0,
            "true_pairs": len(ground_truth),
            "recovered_pairs": 0,
            "pair_recall": 0.0
        }

    candidate_set = set(map(tuple, candidates))
    recovered = len(candidate_set.intersection(ground_truth))
    total = len(ground_truth)
    recall = (recovered / total * 100.0) if total else 0.0

    return {
        "candidate_pairs": len(candidate_set),
        "true_pairs": total,
        "recovered_pairs": recovered,
        "pair_recall": recall
    }
