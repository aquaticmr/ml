import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from unidecode import unidecode
import unicodedata

# ------------------------------------------------------------
# 1. Unicode normalization
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
    "pvt": "private", "pvtltd": "private limited", "ltd": "limited",
    "llc": "limited liability company", "inc": "incorporated",
    "corp": "corporation", "co": "company", "corporation": "corporation",
    "rd": "road", "st": "street", "dr": "drive", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "ln": "lane", "hwy": "highway",
    "ste": "suite", "apt": "apartment", "bldg": "building", "fl": "floor",
    "no": "number", "nr": "near", "opp": "opposite", "mt": "mount",
    "ctr": "center", "pl": "place", "ct": "court", "sq": "square",
}

def expand_abbreviations(text):
    if not text:
        return ""
    tokens = text.split()
    return " ".join(ABBR.get(token, token) for token in tokens)

# ------------------------------------------------------------
# 3. Name normalization
# ------------------------------------------------------------
def normalize_name(x):
    x = normalize_unicode(x)
    x = expand_abbreviations(x)
    return x

def normalize_name_core(x):
    x = normalize_name(x)
    tokens = x.split()
    corporate = {
        "private", "limited", "company", "corporation",
        "incorporated", "llc", "liability",
    }
    tokens = [t for t in tokens if t not in corporate]
    return " ".join(tokens)

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
    nums = re.findall(r"\d+", x)
    return " ".join(nums)

# ------------------------------------------------------------
# 6. Postal code
# ------------------------------------------------------------
def extract_postal(x, country):
    if not x:
        return ""
    if country == "India":
        m = re.findall(r"\b[1-9][0-9]{5}\b", x)
        if m: return m[-1]
    if country == "US":
        m = re.findall(r"\b\d{5}(?:-\d{4})?\b", x)
        if m: return m[-1][:5]
    return ""

# ------------------------------------------------------------
# 7. Create engineered columns
# ------------------------------------------------------------
def engineer_features(df):
    df = df.copy()
    print("Engineering:", len(df), "rows")

    df["name_norm"] = df["business_name"].fillna("").map(normalize_name)
    df["name_core"] = df["business_name"].fillna("").map(normalize_name_core)
    df["address_norm"] = df["business_address"].fillna("").map(normalize_address)
    df["numbers"] = df["address_norm"].map(extract_numbers)
    
    df["postal"] = [
        extract_postal(a, c)
        for a, c in zip(df["address_norm"], df["country"])
    ]

    df["name_tokens"] = df["name_core"].str.split()
    df["name_first"] = df["name_tokens"].str[0].fillna("")
    df["name_last"] = df["name_tokens"].str[-1].fillna("")
    
    df["name_prefix4"] = df["name_core"].str.replace(" ", "", regex=False).str[:4]
    df["name_prefix6"] = df["name_core"].str.replace(" ", "", regex=False).str[:6]
    df["name_chars"] = df["name_core"].str.replace(" ", "", regex=False)

    return df