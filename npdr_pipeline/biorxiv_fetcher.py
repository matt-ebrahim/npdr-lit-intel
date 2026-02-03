"""bioRxiv/medRxiv fetcher with fuzzy title matching.

API documentation: https://api.biorxiv.org/
Rate limits: Be respectful, add delays between requests.
"""

import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Optional
from datetime import datetime, timedelta


# bioRxiv API endpoints
BIORXIV_API_BASE = "https://api.biorxiv.org"
MEDRXIV_API_BASE = "https://api.biorxiv.org"  # Same API, different server param

# Rate limiting: bioRxiv recommends max 1 request per second
REQUEST_DELAY = 1.0  # seconds between requests
MAX_WORKERS = 3  # Parallel workers (conservative to respect rate limits)

# Fuzzy matching threshold (0-1, higher = stricter)
TITLE_MATCH_THRESHOLD = 0.85


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    if not title:
        return ""
    # Lowercase, remove punctuation, normalize whitespace
    normalized = title.lower()
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = ' '.join(normalized.split())
    return normalized


def calculate_similarity(title1: str, title2: str) -> float:
    """Calculate similarity ratio between two titles using SequenceMatcher.

    This is similar to Levenshtein but uses Python's built-in SequenceMatcher
    which is efficient and gives a ratio between 0 and 1.
    """
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def search_biorxiv_by_title(
    title: str,
    server: str = "biorxiv",
    match_threshold: float = TITLE_MATCH_THRESHOLD,
) -> Optional[dict]:
    """Search bioRxiv/medRxiv for a paper by title.

    Args:
        title: Paper title to search for
        server: "biorxiv" or "medrxiv"
        match_threshold: Minimum similarity ratio (0-1) for a match

    Returns:
        Dict with preprint info if found, None otherwise
    """
    if not title:
        return None

    # Extract key terms from title for search
    # bioRxiv content API doesn't have direct title search, so we use date range
    # and filter results. For better results, we search recent preprints.

    # Try the details endpoint with a broad date range
    # This searches the last 2 years of preprints
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    # Use the pubs endpoint which allows searching
    # Format: /pubs/{server}/{interval}
    # We'll search in chunks to find matches

    try:
        # First, try searching via the content API with cursor
        # This searches all preprints and we filter by title
        url = f"{BIORXIV_API_BASE}/details/{server}/{start_date}/{end_date}/0/json"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "collection" not in data:
            return None

        # Search through results for title match
        best_match = None
        best_score = 0

        for paper in data.get("collection", []):
            paper_title = paper.get("title", "")
            score = calculate_similarity(title, paper_title)

            if score > best_score and score >= match_threshold:
                best_score = score
                best_match = paper

        if best_match:
            return {
                "found": True,
                "server": server,
                "doi": best_match.get("doi", ""),
                "title": best_match.get("title", ""),
                "authors": best_match.get("authors", ""),
                "date": best_match.get("date", ""),
                "abstract": best_match.get("abstract", ""),
                "category": best_match.get("category", ""),
                "url": f"https://www.{server}.org/content/{best_match.get('doi', '')}",
                "pdf_url": f"https://www.{server}.org/content/{best_match.get('doi', '')}.full.pdf",
                "match_score": best_score,
                "is_preprint": True,
                "note": "Data from preprint; may differ from published version",
            }

    except requests.RequestException as e:
        print(f"  Warning: bioRxiv API error: {e}")
    except Exception as e:
        print(f"  Warning: bioRxiv search error: {e}")

    return None


def search_biorxiv_by_doi(doi: str) -> Optional[dict]:
    """Search bioRxiv/medRxiv by DOI.

    Args:
        doi: DOI to search for (can be preprint DOI or published DOI)

    Returns:
        Dict with preprint info if found, None otherwise
    """
    if not doi:
        return None

    # Clean DOI
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    for server in ["biorxiv", "medrxiv"]:
        try:
            # Try the published article lookup (links published DOI to preprint)
            url = f"{BIORXIV_API_BASE}/pubs/{server}/{doi}/na/json"

            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "collection" in data and data["collection"]:
                    paper = data["collection"][0]
                    return {
                        "found": True,
                        "server": server,
                        "doi": paper.get("preprint_doi", ""),
                        "title": paper.get("preprint_title", ""),
                        "authors": paper.get("preprint_authors", ""),
                        "date": paper.get("preprint_date", ""),
                        "published_doi": paper.get("published_doi", ""),
                        "url": f"https://www.{server}.org/content/{paper.get('preprint_doi', '')}",
                        "pdf_url": f"https://www.{server}.org/content/{paper.get('preprint_doi', '')}.full.pdf",
                        "match_score": 1.0,
                        "is_preprint": True,
                        "note": "Data from preprint; may differ from published version",
                    }

            time.sleep(REQUEST_DELAY)

        except requests.RequestException:
            pass

    return None


def fetch_biorxiv_full_text(doi: str) -> Optional[str]:
    """Fetch full text from bioRxiv/medRxiv.

    Note: bioRxiv provides HTML/XML for papers. For simplicity, we fetch
    the abstract and any available full text sections.

    Args:
        doi: bioRxiv/medRxiv DOI

    Returns:
        Full text string if available, None otherwise
    """
    if not doi:
        return None

    # Try to get the paper details which include abstract
    for server in ["biorxiv", "medrxiv"]:
        try:
            url = f"{BIORXIV_API_BASE}/details/{server}/{doi}/na/json"

            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "collection" in data and data["collection"]:
                    paper = data["collection"][0]
                    # bioRxiv API doesn't provide full text directly
                    # We can only get abstract from API
                    # For full text, would need to scrape HTML or fetch PDF
                    abstract = paper.get("abstract", "")
                    if abstract:
                        return f"[PREPRINT ABSTRACT]\n{abstract}"

            time.sleep(REQUEST_DELAY)

        except requests.RequestException:
            pass

    return None


def batch_search_biorxiv(
    papers: list,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Search bioRxiv/medRxiv for multiple papers in parallel.

    Args:
        papers: List of dicts with 'pmid', 'title', and optionally 'doi'
        max_workers: Number of parallel workers

    Returns:
        Dict mapping PMID to bioRxiv result (or None if not found)
    """
    results = {}

    print(f"\nSearching bioRxiv/medRxiv for {len(papers)} papers...")
    print(f"Using {max_workers} parallel workers with {REQUEST_DELAY}s delay")

    def search_paper(paper):
        pmid = str(paper.get("pmid", ""))
        title = paper.get("title", "")
        doi = paper.get("doi", "")

        # First try DOI lookup (faster and more accurate)
        if doi:
            result = search_biorxiv_by_doi(doi)
            if result:
                return pmid, result

        # Fall back to title search
        for server in ["medrxiv", "biorxiv"]:  # medRxiv first for medical papers
            result = search_biorxiv_by_title(title, server=server)
            if result:
                return pmid, result
            time.sleep(REQUEST_DELAY)

        return pmid, None

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(search_paper, paper): paper for paper in papers}

        for future in as_completed(futures):
            completed += 1
            pmid, result = future.result()
            results[pmid] = result

            if completed % 10 == 0 or completed == len(papers):
                found = sum(1 for r in results.values() if r is not None)
                print(f"  Progress: {completed}/{len(papers)} searched, {found} found")

    found_count = sum(1 for r in results.values() if r is not None)
    print(f"bioRxiv search complete: {found_count}/{len(papers)} papers found")

    return results
