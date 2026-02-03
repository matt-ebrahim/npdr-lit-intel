"""PubMed search via NCBI E-utilities API."""

import time
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import load_config


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

AI_TERMS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
]


def build_search_query(clinical_term: str, ai_term: str) -> str:
    """Build a PubMed search query combining clinical and AI terms."""
    return f'("{clinical_term}"[Title/Abstract]) AND ("{ai_term}"[Title/Abstract])'


def search_pubmed(query: str, max_results: int = 100, api_key: str = "") -> list[str]:
    """Search PubMed and return list of PMIDs.

    Args:
        query: PubMed search query
        max_results: Maximum number of results to return
        api_key: Optional NCBI API key for higher rate limits

    Returns:
        List of PubMed IDs (PMIDs)
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    response = requests.get(ESEARCH_URL, params=params)
    response.raise_for_status()

    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_article_details(pmids: list[str], api_key: str = "") -> list[dict]:
    """Fetch article details for a list of PMIDs.

    Args:
        pmids: List of PubMed IDs
        api_key: Optional NCBI API key

    Returns:
        List of article dictionaries with metadata
    """
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    response = requests.get(EFETCH_URL, params=params)
    response.raise_for_status()

    # Parse XML response
    root = ET.fromstring(response.content)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue

        pmid_elem = medline.find("PMID")
        article_elem = medline.find("Article")

        if article_elem is None:
            continue

        # Extract title
        title_elem = article_elem.find("ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        # Extract abstract
        abstract_elem = article_elem.find(".//AbstractText")
        abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ""

        # Extract authors
        authors = []
        for author in article_elem.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None and last_name.text:
                name = last_name.text
                if fore_name is not None and fore_name.text:
                    name = f"{fore_name.text} {name}"
                authors.append(name)

        # Extract journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None and journal_elem.text else ""

        # Extract publication date
        pub_date = article_elem.find(".//PubDate")
        year = ""
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            year = year_elem.text if year_elem is not None else ""

        articles.append({
            "pmid": pmid_elem.text if pmid_elem is not None else "",
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_elem.text}/" if pmid_elem is not None else "",
        })

    return articles


def search_clinical_ai_literature(
    clinical_terms: list[str],
    max_results_per_query: int = 50,
    max_workers: int = 3,
) -> list[dict]:
    """Search PubMed for AI/ML literature on clinical terms using parallel requests.

    Args:
        clinical_terms: List of clinical terms (including synonyms)
        max_results_per_query: Max results per individual search
        max_workers: Number of parallel workers (default 3, NCBI rate limit without API key)

    Returns:
        Deduplicated list of article dictionaries
    """
    config = load_config()
    api_key = config.get("ncbi_api_key", "")

    # Rate limit: 3 req/sec without API key, 10 req/sec with API key
    rate_limit_delay = 0.1 if api_key else 0.4

    all_pmids = set()
    total_searches = len(clinical_terms) * len(AI_TERMS)

    print(f"Searching PubMed with {len(clinical_terms)} clinical terms x {len(AI_TERMS)} AI terms = {total_searches} searches...")
    print(f"Using {max_workers} parallel workers\n")

    # Build all search tasks
    search_tasks = [
        (clinical_term, ai_term)
        for clinical_term in clinical_terms
        for ai_term in AI_TERMS
    ]

    def execute_search(task):
        clinical_term, ai_term = task
        query = build_search_query(clinical_term, ai_term)
        try:
            pmids = search_pubmed(query, max_results_per_query, api_key)
            return (clinical_term, ai_term, pmids, None)
        except requests.RequestException as e:
            return (clinical_term, ai_term, [], str(e))

    # Execute searches with rate limiting
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in search_tasks:
            futures.append(executor.submit(execute_search, task))
            time.sleep(rate_limit_delay)  # Rate limit between submissions

        for future in futures:
            completed += 1
            clinical_term, ai_term, pmids, error = future.result()
            if error:
                print(f"  [{completed}/{total_searches}] Warning: '{clinical_term}' + '{ai_term}' failed: {error}")
            else:
                new_pmids = set(pmids) - all_pmids
                if new_pmids:
                    print(f"  [{completed}/{total_searches}] '{clinical_term}' + '{ai_term}': {len(pmids)} results ({len(new_pmids)} new)")
                all_pmids.update(pmids)

    print(f"\nFound {len(all_pmids)} unique articles. Fetching details...")

    # Fetch details in batches of 200 (NCBI limit)
    pmid_list = list(all_pmids)
    all_articles = []

    time.sleep(rate_limit_delay)  # Brief pause before fetching
    for i in range(0, len(pmid_list), 200):
        batch = pmid_list[i:i+200]
        try:
            articles = fetch_article_details(batch, api_key)
            all_articles.extend(articles)
            print(f"  Fetched {len(all_articles)}/{len(pmid_list)} articles...")
        except requests.RequestException as e:
            print(f"  Warning: Failed to fetch batch {i//200 + 1}: {e}")
        time.sleep(rate_limit_delay)

    return all_articles
