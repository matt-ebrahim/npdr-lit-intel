"""PubMed Central full text fetcher.

Uses NCBI E-utilities API to fetch full text from PMC.
Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25499/

OPTIMIZED: Uses persistent file cache to avoid redundant fetches across runs.
"""

import re
import time
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Import cache (with fallback if not available)
try:
    from trial_lit_intel.cache import get_cached, set_cached
    CACHE_ENABLED = True
except ImportError:
    CACHE_ENABLED = False


# NCBI E-utilities endpoints
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"
ELINK_URL = f"{EUTILS_BASE}/elink.fcgi"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"

# Rate limiting: NCBI allows 3 requests/second without API key, 10 with key
REQUEST_DELAY = 0.35  # seconds between requests
MAX_WORKERS = 5  # Parallel workers


def get_pmc_id_from_pmid(pmid: str, api_key: str = None) -> Optional[str]:
    """Convert PubMed ID to PMC ID using elink.

    Args:
        pmid: PubMed ID
        api_key: Optional NCBI API key for higher rate limits

    Returns:
        PMC ID (e.g., "PMC1234567") if available, None otherwise
    """
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        response = requests.get(ELINK_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Navigate the nested structure to find PMC ID
        linksets = data.get("linksets", [])
        if linksets:
            linksetdbs = linksets[0].get("linksetdbs", [])
            for linksetdb in linksetdbs:
                if linksetdb.get("dbto") == "pmc":
                    links = linksetdb.get("links", [])
                    if links:
                        return f"PMC{links[0]}"

    except Exception as e:
        print(f"  Warning: PMC ID lookup failed for PMID {pmid}: {e}")

    return None


def fetch_pmc_full_text(pmc_id: str, api_key: str = None) -> Optional[str]:
    """Fetch full text from PubMed Central.

    Args:
        pmc_id: PMC ID (e.g., "PMC1234567" or just "1234567")
        api_key: Optional NCBI API key

    Returns:
        Full text as string if available, None otherwise
    """
    # Clean PMC ID
    pmc_id = pmc_id.replace("PMC", "")

    params = {
        "db": "pmc",
        "id": pmc_id,
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        response = requests.get(EFETCH_URL, params=params, timeout=60)
        response.raise_for_status()

        # Parse XML and extract text
        full_text = extract_text_from_pmc_xml(response.text)
        return full_text

    except Exception as e:
        print(f"  Warning: PMC fetch failed for {pmc_id}: {e}")

    return None


def extract_text_from_pmc_xml(xml_content: str) -> Optional[str]:
    """Extract readable text from PMC XML.

    Args:
        xml_content: Raw XML from PMC

    Returns:
        Extracted text organized by sections
    """
    try:
        root = ET.fromstring(xml_content)

        sections = []

        # Extract article title
        title_elem = root.find(".//article-title")
        if title_elem is not None:
            title_text = "".join(title_elem.itertext())
            sections.append(f"TITLE: {title_text.strip()}")

        # Extract abstract
        abstract_elem = root.find(".//abstract")
        if abstract_elem is not None:
            abstract_text = "".join(abstract_elem.itertext())
            abstract_text = re.sub(r'\s+', ' ', abstract_text).strip()
            sections.append(f"\nABSTRACT:\n{abstract_text}")

        # Extract body sections
        body = root.find(".//body")
        if body is not None:
            for sec in body.findall(".//sec"):
                # Get section title
                sec_title = sec.find("title")
                if sec_title is not None:
                    sec_title_text = "".join(sec_title.itertext()).strip().upper()
                    sections.append(f"\n{sec_title_text}:")

                # Get paragraphs
                for p in sec.findall(".//p"):
                    p_text = "".join(p.itertext())
                    p_text = re.sub(r'\s+', ' ', p_text).strip()
                    if p_text:
                        sections.append(p_text)

        # Extract tables (just captions for now)
        for table_wrap in root.findall(".//table-wrap"):
            caption = table_wrap.find(".//caption")
            if caption is not None:
                caption_text = "".join(caption.itertext())
                caption_text = re.sub(r'\s+', ' ', caption_text).strip()
                sections.append(f"\n[TABLE]: {caption_text}")

        # Extract figure captions
        for fig in root.findall(".//fig"):
            caption = fig.find(".//caption")
            if caption is not None:
                caption_text = "".join(caption.itertext())
                caption_text = re.sub(r'\s+', ' ', caption_text).strip()
                sections.append(f"\n[FIGURE]: {caption_text}")

        if sections:
            return "\n".join(sections)

    except ET.ParseError as e:
        print(f"  Warning: XML parse error: {e}")

    return None


def fetch_full_text_for_pmid(pmid: str, api_key: str = None) -> Optional[dict]:
    """Fetch full text for a PubMed ID if available in PMC.

    Args:
        pmid: PubMed ID
        api_key: Optional NCBI API key

    Returns:
        Dict with full_text and metadata, or None if not in PMC
    """
    # First, check if paper is in PMC
    pmc_id = get_pmc_id_from_pmid(pmid, api_key)

    if not pmc_id:
        return None

    # Fetch full text
    full_text = fetch_pmc_full_text(pmc_id, api_key)

    if full_text:
        return {
            "pmid": pmid,
            "pmc_id": pmc_id,
            "full_text": full_text,
            "source": "PMC",
            "is_preprint": False,
        }

    return None


def batch_fetch_pmc_full_text(
    pmids: list,
    api_key: str = None,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Fetch full text from PMC for multiple PMIDs in parallel.

    OPTIMIZED:
    1. Uses persistent cache to skip already-fetched papers
    2. Uses batch availability check first, then only fetches papers that exist in PMC
    This reduces API calls from 2*N to 1 + N_found (typically 95% reduction).

    Args:
        pmids: List of PubMed IDs
        api_key: Optional NCBI API key
        max_workers: Number of parallel workers

    Returns:
        Dict mapping PMID to full text result (or None if not in PMC)
    """
    results = {str(pmid): None for pmid in pmids}
    pmids_to_check = []

    # Step 0: Check persistent cache first
    if CACHE_ENABLED:
        cached_count = 0
        for pmid in pmids:
            pmid_str = str(pmid)
            cached = get_cached(pmid_str, "pmc")
            if cached is not None:
                results[pmid_str] = cached if cached != "NOT_IN_PMC" else None
                cached_count += 1
            else:
                pmids_to_check.append(pmid)

        if cached_count > 0:
            print(f"\nUsing {cached_count} cached PMC results")

        if not pmids_to_check:
            found_count = sum(1 for r in results.values() if r is not None)
            print(f"All {len(pmids)} papers found in cache ({found_count} have full text)")
            return results
    else:
        pmids_to_check = list(pmids)

    print(f"\nChecking PMC availability for {len(pmids_to_check)} papers (batch mode)...")

    # Step 1: Batch check which PMIDs have PMC full text (1 API call per 200 PMIDs)
    pmc_ids = check_pmc_availability(pmids_to_check, api_key)

    # Filter to only papers with PMC IDs
    pmids_with_pmc = [(pmid, pmc_id) for pmid, pmc_id in pmc_ids.items() if pmc_id]

    # Cache papers NOT in PMC to avoid rechecking
    if CACHE_ENABLED:
        for pmid, pmc_id in pmc_ids.items():
            if not pmc_id:
                set_cached(str(pmid), "pmc", "NOT_IN_PMC")

    if not pmids_with_pmc:
        print(f"No papers found in PMC")
        return results

    print(f"Found {len(pmids_with_pmc)}/{len(pmids_to_check)} papers in PMC, fetching full text...")
    print(f"Using {max_workers} parallel workers")

    # Step 2: Fetch full text only for papers that exist in PMC (parallel)
    def fetch_single(pmid_pmc_tuple):
        pmid, pmc_id = pmid_pmc_tuple
        time.sleep(REQUEST_DELAY)  # Rate limiting
        full_text = fetch_pmc_full_text(pmc_id, api_key)
        if full_text:
            result = {
                "pmid": pmid,
                "pmc_id": pmc_id,
                "full_text": full_text,
                "source": "PMC",
                "is_preprint": False,
            }
            # Cache successful fetch
            if CACHE_ENABLED:
                set_cached(str(pmid), "pmc", result)
            return pmid, result
        return pmid, None

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single, item): item for item in pmids_with_pmc}

        for future in as_completed(futures):
            completed += 1
            pmid, result = future.result()
            results[pmid] = result

            if completed % 5 == 0 or completed == len(pmids_with_pmc):
                found = sum(1 for r in results.values() if r is not None)
                print(f"  Progress: {completed}/{len(pmids_with_pmc)} fetched, {found} successful")

    found_count = sum(1 for r in results.values() if r is not None)
    print(f"PMC fetch complete: {found_count}/{len(pmids)} papers have full text")

    return results


def check_pmc_availability(pmids: list, api_key: str = None) -> dict:
    """Quick check which PMIDs have full text in PMC.

    Args:
        pmids: List of PubMed IDs
        api_key: Optional NCBI API key

    Returns:
        Dict mapping PMID to PMC ID (or None if not in PMC)
    """
    # Use batch elink for efficiency
    results = {}

    # Process in batches of 200 (NCBI limit)
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]

        params = {
            "dbfrom": "pubmed",
            "db": "pmc",
            "id": ",".join(str(p) for p in batch),
            "retmode": "json",
        }
        if api_key:
            params["api_key"] = api_key

        try:
            response = requests.get(ELINK_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Parse results
            for linkset in data.get("linksets", []):
                pmid = str(linkset.get("ids", [None])[0])
                pmc_id = None

                for linksetdb in linkset.get("linksetdbs", []):
                    if linksetdb.get("dbto") == "pmc":
                        links = linksetdb.get("links", [])
                        if links:
                            pmc_id = f"PMC{links[0]}"
                            break

                results[pmid] = pmc_id

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"  Warning: PMC availability check failed: {e}")
            # Mark batch as unchecked
            for pmid in batch:
                results[str(pmid)] = None

    return results
