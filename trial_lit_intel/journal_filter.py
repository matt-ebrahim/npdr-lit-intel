"""Journal quality filtering using whitelist of top medical AI journals."""

# Top 20 Medical AI Journals - curated from impact factors and relevance
# Sources: Scimago, Clarivate JCR, Nature, RSNA, Lancet
#
# Categories:
# - Dedicated Medical AI/Digital Health journals
# - High-impact general medical journals that publish AI research
# - Top specialty journals (radiology, ophthalmology, cardiology)

JOURNAL_WHITELIST = {
    # === Dedicated Medical AI / Digital Health Journals ===

    # npj Digital Medicine (Nature) - IF ~15.1, Q1
    "npj digital medicine": {"tier": 1, "category": "digital_health"},
    "npj digit med": {"tier": 1, "category": "digital_health"},

    # The Lancet Digital Health - High impact, Elsevier
    "lancet digital health": {"tier": 1, "category": "digital_health"},
    "the lancet digital health": {"tier": 1, "category": "digital_health"},

    # Radiology: Artificial Intelligence (RSNA) - IF ~13.2
    "radiology: artificial intelligence": {"tier": 1, "category": "radiology_ai"},
    "radiology artificial intelligence": {"tier": 1, "category": "radiology_ai"},
    "radiol artif intell": {"tier": 1, "category": "radiology_ai"},

    # NEJM AI - New England Journal of Medicine AI
    "nejm ai": {"tier": 1, "category": "medical_ai"},
    "new england journal of medicine ai": {"tier": 1, "category": "medical_ai"},

    # Nature Medicine - IF ~80+, covers digital medicine/AI
    "nature medicine": {"tier": 1, "category": "general_medical"},
    "nat med": {"tier": 1, "category": "general_medical"},

    # Artificial Intelligence in Medicine (Elsevier) - IF ~6-8, Q1
    "artificial intelligence in medicine": {"tier": 2, "category": "medical_ai"},
    "artif intell med": {"tier": 2, "category": "medical_ai"},

    # IEEE Journal of Biomedical and Health Informatics - IF ~7
    "ieee journal of biomedical and health informatics": {"tier": 2, "category": "health_informatics"},
    "ieee j biomed health inform": {"tier": 2, "category": "health_informatics"},
    "j biomed health inform": {"tier": 2, "category": "health_informatics"},

    # JMIR Medical Informatics / JMIR AI
    "jmir medical informatics": {"tier": 2, "category": "health_informatics"},
    "jmir ai": {"tier": 2, "category": "medical_ai"},
    "journal of medical internet research": {"tier": 2, "category": "health_informatics"},
    "j med internet res": {"tier": 2, "category": "health_informatics"},

    # PLOS Digital Health
    "plos digital health": {"tier": 2, "category": "digital_health"},

    # Journal of Medical Artificial Intelligence
    "journal of medical artificial intelligence": {"tier": 2, "category": "medical_ai"},

    # === High-Impact General Medical/Science Journals ===

    # Nature Communications - IF ~16
    "nature communications": {"tier": 1, "category": "general_science"},
    "nat commun": {"tier": 1, "category": "general_science"},

    # JAMA Network Open
    "jama network open": {"tier": 1, "category": "general_medical"},

    # JAMA
    "jama": {"tier": 1, "category": "general_medical"},
    "jama-journal of the american medical association": {"tier": 1, "category": "general_medical"},

    # Cell
    "cell": {"tier": 1, "category": "general_science"},

    # Science Translational Medicine
    "science translational medicine": {"tier": 1, "category": "general_science"},
    "sci transl med": {"tier": 1, "category": "general_science"},

    # === Top Specialty Journals (Radiology) ===

    # Radiology (RSNA) - IF ~12
    "radiology": {"tier": 1, "category": "radiology"},

    # European Radiology - IF ~5
    "european radiology": {"tier": 2, "category": "radiology"},
    "eur radiol": {"tier": 2, "category": "radiology"},

    # Medical Image Analysis - IF ~10
    "medical image analysis": {"tier": 1, "category": "medical_imaging"},
    "med image anal": {"tier": 1, "category": "medical_imaging"},

    # === Top Specialty Journals (Ophthalmology) ===

    # Ophthalmology (AAO) - IF ~13
    "ophthalmology": {"tier": 1, "category": "ophthalmology"},

    # JAMA Ophthalmology - IF ~8
    "jama ophthalmology": {"tier": 1, "category": "ophthalmology"},
    "jama ophthalmol": {"tier": 1, "category": "ophthalmology"},

    # British Journal of Ophthalmology - IF ~4
    "british journal of ophthalmology": {"tier": 2, "category": "ophthalmology"},
    "br j ophthalmol": {"tier": 2, "category": "ophthalmology"},

    # American Journal of Ophthalmology - IF ~4
    "american journal of ophthalmology": {"tier": 2, "category": "ophthalmology"},
    "am j ophthalmol": {"tier": 2, "category": "ophthalmology"},

    # Investigative Ophthalmology & Visual Science - IF ~5
    "investigative ophthalmology & visual science": {"tier": 2, "category": "ophthalmology"},
    "invest ophthalmol vis sci": {"tier": 2, "category": "ophthalmology"},

    # === Top Specialty Journals (Cardiology) ===

    # Circulation - IF ~35
    "circulation": {"tier": 1, "category": "cardiology"},

    # European Heart Journal - IF ~35
    "european heart journal": {"tier": 1, "category": "cardiology"},
    "eur heart j": {"tier": 1, "category": "cardiology"},

    # JACC (Journal of the American College of Cardiology) - IF ~21
    "journal of the american college of cardiology": {"tier": 1, "category": "cardiology"},
    "jacc": {"tier": 1, "category": "cardiology"},
    "j am coll cardiol": {"tier": 1, "category": "cardiology"},

    # === Other High-Impact Specialty Journals ===

    # The Lancet - IF ~160
    "lancet": {"tier": 1, "category": "general_medical"},
    "the lancet": {"tier": 1, "category": "general_medical"},

    # New England Journal of Medicine - IF ~150
    "new england journal of medicine": {"tier": 1, "category": "general_medical"},
    "n engl j med": {"tier": 1, "category": "general_medical"},

    # Lancet Oncology - IF ~50
    "lancet oncology": {"tier": 1, "category": "oncology"},
    "the lancet oncology": {"tier": 1, "category": "oncology"},

    # Scientific Reports (Nature) - IF ~4 but high volume AI research
    "scientific reports": {"tier": 2, "category": "general_science"},
    "sci rep": {"tier": 2, "category": "general_science"},
}


def normalize_journal_name(journal_name: str) -> str:
    """Normalize journal name for matching."""
    if not journal_name:
        return ""
    # Lowercase, remove extra spaces, common punctuation
    normalized = journal_name.lower().strip()
    normalized = normalized.replace(".", "").replace(",", "").replace(":", " ")
    normalized = " ".join(normalized.split())  # Normalize whitespace
    return normalized


def is_high_quality_journal(journal_name: str, tier_threshold: int = 2) -> bool:
    """Check if journal is in the whitelist.

    Args:
        journal_name: Name of the journal
        tier_threshold: Maximum tier to include (1=top tier only, 2=include second tier)

    Returns:
        True if journal is in whitelist at or above tier threshold
    """
    normalized = normalize_journal_name(journal_name)

    if normalized in JOURNAL_WHITELIST:
        return JOURNAL_WHITELIST[normalized]["tier"] <= tier_threshold

    # Try partial matching for journals with slight name variations
    for whitelist_name, info in JOURNAL_WHITELIST.items():
        if info["tier"] <= tier_threshold:
            # Check if whitelist name is contained in the journal name or vice versa
            if whitelist_name in normalized or normalized in whitelist_name:
                return True

    return False


def get_journal_info(journal_name: str):
    """Get journal tier and category if in whitelist."""
    normalized = normalize_journal_name(journal_name)

    if normalized in JOURNAL_WHITELIST:
        return JOURNAL_WHITELIST[normalized]

    # Try partial matching
    for whitelist_name, info in JOURNAL_WHITELIST.items():
        if whitelist_name in normalized or normalized in whitelist_name:
            return info

    return None


def filter_by_journal_quality(
    articles: list[dict],
    tier_threshold: int = 2,
) -> list[dict]:
    """Filter articles to only include those from high-quality journals.

    Args:
        articles: List of article dicts with 'journal' field
        tier_threshold: Maximum tier to include (1=top tier only, 2=include second tier)

    Returns:
        Filtered list of articles from whitelisted journals
    """
    if not articles:
        return []

    filtered = []
    journal_stats = {"matched": 0, "unmatched": 0, "journals_found": set()}

    for article in articles:
        journal = article.get("journal", "")
        if is_high_quality_journal(journal, tier_threshold):
            article_copy = article.copy()
            journal_info = get_journal_info(journal)
            if journal_info:
                article_copy["journal_tier"] = journal_info["tier"]
                article_copy["journal_category"] = journal_info["category"]
            filtered.append(article_copy)
            journal_stats["matched"] += 1
            journal_stats["journals_found"].add(journal)
        else:
            journal_stats["unmatched"] += 1

    print(f"\nJournal quality filter:")
    print(f"  Matched: {journal_stats['matched']} articles from {len(journal_stats['journals_found'])} journals")
    print(f"  Filtered out: {journal_stats['unmatched']} articles from non-whitelisted journals")

    return filtered


def get_whitelist_summary() -> dict:
    """Get summary of journals in whitelist by category and tier."""
    summary = {
        "total_entries": len(JOURNAL_WHITELIST),
        "by_tier": {1: [], 2: []},
        "by_category": {},
    }

    seen = set()  # Track unique journals (avoid counting aliases)

    for name, info in JOURNAL_WHITELIST.items():
        # Use first 3 words as key to detect aliases
        key = " ".join(name.split()[:3])
        if key not in seen:
            seen.add(key)
            tier = info["tier"]
            category = info["category"]

            summary["by_tier"][tier].append(name)

            if category not in summary["by_category"]:
                summary["by_category"][category] = []
            summary["by_category"][category].append(name)

    return summary
