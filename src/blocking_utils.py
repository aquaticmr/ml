"""
blocking_utils.py - Modular normalization, transliteration, blocking key generation,
and Parquet writing for Amazon ML Entity Resolution.
"""

from pathlib import Path
import os
import re
import gc
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from unidecode import unidecode
import pyarrow as pa
import pyarrow.parquet as pq

NULL_LIKE = {
    "",
    "null",
    "none",
    "nan",
    "n/a",
}

# ------------------------------------------------------------
# Script detection
# ------------------------------------------------------------

SCRIPT_PATTERNS = {
    "DEVANAGARI": re.compile(r"[\u0900-\u097F]"),
    "BENGALI": re.compile(r"[\u0980-\u09FF]"),
    "GURMUKHI": re.compile(r"[\u0A00-\u0A7F]"),
    "GUJARATI": re.compile(r"[\u0A80-\u0AFF]"),
    "ORIYA": re.compile(r"[\u0B00-\u0B7F]"),
    "TAMIL": re.compile(r"[\u0B80-\u0BFF]"),
    "TELUGU": re.compile(r"[\u0C00-\u0C7F]"),
    "KANNADA": re.compile(r"[\u0C80-\u0CFF]"),
    "MALAYALAM": re.compile(r"[\u0D00-\u0D7F]"),
    "ARABIC": re.compile(r"[\u0600-\u06FF]"),
    "CYRILLIC": re.compile(r"[\u0400-\u04FF]"),
}


def detect_script_fast(text):
    if pd.isna(text):
        return "EMPTY"

    text = str(text).strip()

    if not text:
        return "EMPTY"

    if text.isascii():
        if any(ch.isalpha() for ch in text):
            return "LATIN"
        return "OTHER"

    scripts = []

    for script, pattern in SCRIPT_PATTERNS.items():
        if pattern.search(text):
            scripts.append(script)

    # Detect Latin characters in mixed strings
    if any(ch.isascii() and ch.isalpha() for ch in text):
        scripts.append("LATIN")

    if not scripts:
        return "OTHER"

    scripts = set(scripts)

    if len(scripts) == 1:
        return next(iter(scripts))

    return "MIXED"


# ------------------------------------------------------------
# Unicode normalization
# ------------------------------------------------------------

def clean_unicode(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    # Make punctuation act like separators
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        token
        for token in text.split()
        if token not in NULL_LIKE
    ]

    return " ".join(tokens)


# ------------------------------------------------------------
# Transliteration
# ------------------------------------------------------------

def clean_transliterated(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    text = text.casefold()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        token
        for token in text.split()
        if token not in NULL_LIKE
    ]

    return " ".join(tokens)


# ------------------------------------------------------------
# Corporate abbreviations
# ------------------------------------------------------------

ABBREVIATIONS = {
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "ltd": "limited",
    "pvt": "private",

    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",

    "apt": "apartment",
    "fl": "floor",
    "ste": "suite",
}


def expand_abbreviations(text):
    tokens = text.split()
    return " ".join(ABBREVIATIONS.get(token, token) for token in tokens)


# ------------------------------------------------------------
# Legal suffix removal
# ------------------------------------------------------------

LEGAL_SINGLE = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "pvt",
    "private",
    "llc",
}

LEGAL_PHRASES = [
    ("limited", "liability", "company"),
    ("private", "limited"),
]


def strip_legal_suffix(tokens):
    tokens = list(tokens)
    changed = True

    while changed and tokens:
        changed = False

        # Multi-token suffixes
        for phrase in LEGAL_PHRASES:
            n = len(phrase)
            if len(tokens) > n and tuple(tokens[-n:]) == phrase:
                tokens = tokens[:-n]
                changed = True
                break

        if changed:
            continue

        # Single-token suffix
        if len(tokens) > 1 and tokens[-1] in LEGAL_SINGLE:
            tokens.pop()
            changed = True

    return tokens


# ------------------------------------------------------------
# Name core & compact
# ------------------------------------------------------------

def name_core(name_translit):
    tokens = name_translit.split()
    if not tokens:
        return ""

    core = strip_legal_suffix(tokens)
    if not core:
        core = tokens

    return " ".join(core)


def compact_alnum(text):
    return re.sub(r"[^a-z0-9]", "", text)


# ------------------------------------------------------------
# Name token pair
# ------------------------------------------------------------

def longest_token_pair(name_core_text):
    tokens = [
        token
        for token in name_core_text.split()
        if len(token) >= 3 and not token.isdigit()
    ]

    if not tokens:
        return ""

    tokens = sorted(tokens, key=lambda x: (-len(x), x))
    selected = sorted(tokens[:2])
    return "|".join(selected)


# ------------------------------------------------------------
# Postal code
# ------------------------------------------------------------

INDIA_PIN_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
US_ZIP_RE = re.compile(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)")


def extract_postal(address, country):
    if pd.isna(address):
        return ""

    address = str(address)
    country = "" if pd.isna(country) else str(country).strip().upper()

    if country == "INDIA":
        matches = INDIA_PIN_RE.findall(address)
    elif country == "US":
        matches = US_ZIP_RE.findall(address)
    else:
        matches = []

    return matches[-1] if matches else ""


# ------------------------------------------------------------
# Main record transformation
# ------------------------------------------------------------

def build_block_features(df):
    out = pd.DataFrame()

    out["entity_id"] = (
        df["entity_id"]
        .fillna("")
        .astype("string")
    )

    out["country"] = (
        df["country"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.upper()
    )

    name_raw = (
        df["business_name"]
        .fillna("")
        .astype("string")
    )

    address_raw = (
        df["business_address"]
        .fillna("")
        .astype("string")
    )

    # Scripts
    out["name_script"] = name_raw.map(detect_script_fast)
    out["address_script"] = address_raw.map(detect_script_fast)

    # Normalized Unicode
    name_norm = name_raw.map(clean_unicode)
    address_norm = address_raw.map(clean_unicode)

    # Transliteration
    name_translit = name_raw.map(clean_transliterated)
    address_translit = address_raw.map(clean_transliterated)

    out["name_translit"] = name_translit
    out["address_translit"] = address_translit

    # Expanded versions
    name_expanded = name_translit.map(expand_abbreviations)
    address_expanded = address_translit.map(expand_abbreviations)

    out["name_expanded"] = name_expanded
    out["address_expanded"] = address_expanded

    # Core business name
    core = name_expanded.map(name_core)
    out["name_core"] = core

    compact = core.map(compact_alnum)
    out["name_compact"] = compact

    # Name blocking keys
    out["name_prefix4"] = compact.str[:4]
    out["name_prefix6"] = compact.str[:6]
    out["name_exact"] = core
    out["name_token_pair"] = core.map(longest_token_pair)

    # Postal code
    out["postal_code"] = [
        extract_postal(address, country)
        for address, country in zip(address_raw, out["country"])
    ]

    # Actual blocking keys
    country = out["country"].fillna("")

    out["block_postal_name4"] = (
        country
        + "|"
        + out["postal_code"].fillna("")
        + "|"
        + out["name_prefix4"].fillna("")
    )

    out["block_name_prefix6"] = (
        country
        + "|"
        + out["name_prefix6"].fillna("")
    )

    out["block_name_exact"] = (
        country
        + "|"
        + out["name_exact"].fillna("")
    )

    out["block_token_pair"] = (
        country
        + "|"
        + out["name_token_pair"].fillna("")
    )

    out["block_postal"] = (
        country
        + "|"
        + out["postal_code"].fillna("")
    )

    # Empty blocking keys should not be usable.
    for col in [
        "block_postal_name4",
        "block_name_prefix6",
        "block_name_exact",
        "block_token_pair",
        "block_postal"
    ]:
        out.loc[out[col].str.endswith("|"), col] = ""
        out.loc[out[col].isin(["", "|"]), col] = ""

    return out


def write_key_parquet(
    input_path,
    output_path,
    chunk_size=250_000
):
    input_path = str(input_path)
    output_path = str(output_path)

    if os.path.exists(output_path):
        os.remove(output_path)

    writer = None
    total_rows = 0

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]

    try:
        reader = pd.read_csv(
            input_path,
            sep="\t",
            dtype="string",
            usecols=usecols,
            chunksize=chunk_size,
            keep_default_na=True
        )

        for chunk in tqdm(
            reader,
            desc=f"Building {Path(output_path).name}"
        ):
            features = build_block_features(chunk)

            table = pa.Table.from_pandas(
                features,
                preserve_index=False
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd"
                )

            writer.write_table(table)
            total_rows += len(chunk)

            del chunk
            del features
            del table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    print(f"Written {total_rows:,} rows -> {output_path}")
