"""
Stage 1: Text Normalization for Entity Resolution.
Transforms noisy business names and addresses into clean, comparable forms.
"""
import re
import unicodedata


# ─── Legal Suffix Standardization ────────────────────────────────────────────
# Maps all known variants to a single canonical form.
# Order matters: longer patterns first to avoid partial replacements.
LEGAL_SUFFIX_MAP = [
    # LLC variants
    (r'\bl\.?\s*l\.?\s*c\.?\b', 'llc'),
    # LLP variants
    (r'\bl\.?\s*l\.?\s*p\.?\b', 'llp'),
    # Private Limited variants
    (r'\bpvt\.?\s*ltd\.?\b', 'private limited'),
    (r'\bprivate\s+ltd\.?\b', 'private limited'),
    (r'\bpvt\.?\s*limited\b', 'private limited'),
    # Limited
    (r'\bltd\.?\b', 'limited'),
    # Corporation
    (r'\bcorp\.?\b', 'corporation'),
    # Incorporated
    (r'\binc\.?\b', 'incorporated'),
    # Company
    (r'\bco\.?\b', 'company'),
    # Public Company
    (r'\bp\.?\s*c\.?\b(?!\w)', 'pc'),
    # DBA
    (r'\bd/?b/?a\b', 'dba'),
    # French legal suffixes
    (r'\bs\.?\s*a\.?\s*r\.?\s*l\.?\b', 'sarl'),
    (r'\bs\.?\s*a\.?\s*s\.?\s*u\.?\b', 'sasu'),
    (r'\bs\.?\s*a\.?\s*s\.?\b', 'sas'),
    (r'\be\.?\s*u\.?\s*r\.?\s*l\.?\b', 'eurl'),
    (r'\bs\.?\s*c\.?\s*i\.?\b', 'sci'),
    # Indian honorifics / prefixes
    (r'\bm\s*/\s*s\.?\b', ''),
    (r'\bm\s+s\b', ''),
    (r'\bshree\b', 'sri'),
    (r'\bshri\b', 'sri'),
]

# ─── Address Abbreviation Expansion ──────────────────────────────────────────
ADDRESS_ABBREV_MAP = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'cir': 'circle', 'pkwy': 'parkway', 'hwy': 'highway', 'sq': 'square',
    'ter': 'terrace', 'trl': 'trail', 'wy': 'way', 'expy': 'expressway',
    'apt': 'apartment', 'ste': 'suite', 'bldg': 'building', 'fl': 'floor',
    'dept': 'department', 'rm': 'room', 'spc': 'space', 'ofc': 'office',
    # French address terms
    'bd': 'boulevard', 'bld': 'boulevard', 'av': 'avenue', 'r': 'rue',
    'imp': 'impasse', 'all': 'allee', 'ch': 'chemin', 'rte': 'route',
    # Directions
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    # US state abbreviations (common ones that appear in addresses)
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
}

# ─── Noise Words to Remove ──────────────────────────────────────────────────
NOISE_WORDS = {'null', 'nan', 'none', 'n/a', 'na', 'n a', 'not available',
               'not applicable', 'unknown', 'unspecified'}

# Pre-compile regex patterns for performance
_LEGAL_PATTERNS = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in LEGAL_SUFFIX_MAP]
_ADDR_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(ADDRESS_ABBREV_MAP.keys(), key=len, reverse=True)) + r')\b'
)
_PUNCT_PATTERN = re.compile(r'[^\w\s]')
_MULTI_SPACE = re.compile(r'\s+')
_NUMBER_PATTERN = re.compile(r'\b\d+\b')

# Ordinal fix: "1th" → "1st", "2th" → "2nd", etc.
_ORDINAL_FIX = re.compile(r'\b(\d+)(?:th|TH)\b')


import anyascii


def strip_accents(text: str) -> str:
    """Remove diacritical marks (accents) via Unicode NFKD decomposition."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def normalize_text(text: str) -> str:
    """
    Core normalization: transliterate Unicode to Latin, lowercase,
    strip accents, remove punctuation, collapse whitespace.
    Used for both names and addresses.
    """
    if not isinstance(text, str) or not text.strip():
        return ''

    # Check for noise words
    if text.strip().lower() in NOISE_WORDS:
        return ''

    # Transliterate any non-Latin scripts (Hindi, Telugu, Tamil, Arabic, etc.) to Latin
    text = anyascii.anyascii(text)

    # Strip accents and diacritics
    text = strip_accents(text)

    # Lowercase
    text = text.lower()

    # Replace & with 'and'
    text = text.replace('&', ' and ')
    text = text.replace('+', ' and ')

    # Remove punctuation (keep alphanumeric + spaces)
    text = _PUNCT_PATTERN.sub(' ', text)

    # Collapse whitespace
    text = _MULTI_SPACE.sub(' ', text).strip()

    return text


def normalize_business_name(name: str) -> str:
    """
    Normalize a business name: standard normalization + legal suffix mapping.
    """
    text = normalize_text(name)
    if not text:
        return ''

    # Apply legal suffix standardization
    for pattern, replacement in _LEGAL_PATTERNS:
        text = pattern.sub(replacement, text)

    # Re-collapse whitespace
    text = _MULTI_SPACE.sub(' ', text).strip()

    return text


def normalize_address(address: str) -> str:
    """
    Normalize an address: standard normalization + address abbreviation expansion.
    """
    text = normalize_text(address)
    if not text:
        return ''

    # Fix ordinals: 1th → 1st, etc.
    text = _ORDINAL_FIX.sub(lambda m: m.group(1) + _ordinal_suffix(int(m.group(1))), text)

    # Expand address abbreviations
    # Note: We do NOT expand state abbreviations here to avoid ambiguity
    # (e.g., 'in' could be Indiana or a preposition). Keep it simple.
    # Only expand unambiguous street-type abbreviations.
    def _expand_addr(match):
        word = match.group(1).lower()
        # Only expand known unambiguous street abbreviations
        if word in {'rd', 'ave', 'blvd', 'dr', 'ln', 'ct', 'pl', 'cir',
                     'pkwy', 'hwy', 'sq', 'ter', 'trl', 'expy',
                     'apt', 'ste', 'bldg'}:
            return ADDRESS_ABBREV_MAP.get(word, word)
        return word

    text = _ADDR_PATTERN.sub(_expand_addr, text)

    # Re-collapse whitespace
    text = _MULTI_SPACE.sub(' ', text).strip()

    return text


def extract_numeric_tokens(text: str) -> list:
    """Extract all numeric tokens from a string (street numbers, PIN codes, etc.)."""
    if not isinstance(text, str):
        return []
    return _NUMBER_PATTERN.findall(text)


def extract_name_tokens(normalized_name: str) -> set:
    """Extract the set of non-empty tokens from a normalized name."""
    if not normalized_name:
        return set()
    return set(normalized_name.split())


def _ordinal_suffix(n: int) -> str:
    """Return the ordinal suffix for a number."""
    if 11 <= (n % 100) <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


# ─── Convenience: normalize a full record ────────────────────────────────────
def normalize_record(entity_id: str, business_name: str, business_address: str,
                     country: str) -> dict:
    """
    Normalize a single record, returning a dict with all normalized fields.
    """
    norm_name = normalize_business_name(business_name)
    norm_addr = normalize_address(business_address)
    return {
        'entity_id': entity_id,
        'norm_name': norm_name,
        'norm_addr': norm_addr,
        'name_tokens': extract_name_tokens(norm_name),
        'addr_numbers': extract_numeric_tokens(str(business_address)),
        'country': str(country).strip() if isinstance(country, str) else str(country),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    test_cases = [
        ("QH Vendome, L.L.C.", "Katy, TX, 21342 Bending Green Way"),
        ("Modern Packaging Global PC", "8935 Georgetown Pike, FAIRFAX COUNTY, VA"),
        ("The Blue Netw0rk Corporation", "177 1TH AVENUE, NASHVILLE, TN"),
        ("Moran Seaffng LLC", "1902 Shamrock Rd, Dothan, Alabama"),
        ("Porur Vyápar Private Limited", "NO.137, GANDHI STREET, 2ND MAIN ROAD SUBHASHRI NAGAR"),
        ("Pvt. EFS Print Ventures Ltd.", "Door No 183, 41St Cross, Bengaluru, ಕರ್ನಾಟಕ"),
        ("LLC Moran Staffing", "1902 Shamrock Rd, Alabama, Dothan"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI"),
    ]

    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 80)
    print("NORMALIZATION TEST")
    print("=" * 80)
    for name, addr in test_cases:
        norm = normalize_record("test", name, addr, "US")
        print(f"\n  Name   : {name!r}")
        print(f"  → Norm : {norm['norm_name']!r}")
        print(f"  → Tokens: {norm['name_tokens']}")
        print(f"  Addr   : {addr!r}")
        print(f"  → Norm : {norm['norm_addr']!r}")
        print(f"  → Nums : {norm['addr_numbers']}")
