"""MeSH-based PubMed search using snek for ontology mapping.

This module provides a more technical approach to term expansion by:
1. Using snek to map clinical terms to MeSH ontology IDs
2. Querying PubMed via Snowflake MESH_HEADINGS table
3. Still filtering for AI/ML terms in title/abstract
"""

import os
import requests
from functools import lru_cache
from typing import Optional
import pandas as pd
from .snowflake_search import get_snowflake_service, escape_sql_string, AI_TERMS
from .config import get_secret

# Snek API endpoint
SNEK_API_URL = "https://snek.application.formation.bio"

# In-memory cache for MeSH terms
_mesh_cache = {}


def get_snek_api_key() -> str:
    """Get snek API key from environment or secrets."""
    return get_secret("SNEK_API_KEY") or os.getenv("SNEK_API_KEY", "")


def get_mesh_terms_from_snek(condition: str, api_key: str = None) -> list[dict]:
    """Use snek to get MeSH terms for a clinical condition.

    OPTIMIZED: Results are cached in memory to avoid redundant API calls.

    Args:
        condition: Clinical condition name (e.g., "diabetic retinopathy")
        api_key: Snek API Bearer token (uses env var if not provided)

    Returns:
        List of dicts with 'mesh_id', 'mesh_name', 'labels', 'confidence'
    """
    # Check cache first
    cache_key = condition.lower().strip()
    if cache_key in _mesh_cache:
        print(f"  Using cached MeSH terms for '{condition}'")
        return _mesh_cache[cache_key]

    api_key = api_key or get_snek_api_key()

    if not api_key:
        print("  Warning: No SNEK_API_KEY configured, skipping snek lookup")
        return []

    mesh_terms = []
    all_labels = {}  # Track all ontology labels by type

    try:
        # POST to /nerd/entities/ with condition as JSON string body
        response = requests.post(
            f"{SNEK_API_URL}/nerd/entities/",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=condition,  # Send as JSON string
            timeout=30,
        )

        if response.status_code == 200:
            entities = response.json()

            for entity in entities:
                text = entity.get("text", condition)
                labels = entity.get("labels", [])
                confidence = entity.get("confidence", 0)

                # Extract MeSH IDs from labels
                for label in labels:
                    if label.startswith("MESH:"):
                        mesh_id = label.replace("MESH:", "")
                        mesh_terms.append({
                            "mesh_id": mesh_id,
                            "mesh_name": text,
                            "source": "snek_nerd",
                            "confidence": confidence,
                            "all_labels": labels,
                        })

                # Also store all labels for reference
                for label in labels:
                    prefix = label.split(":")[0] if ":" in label else "OTHER"
                    if prefix not in all_labels:
                        all_labels[prefix] = []
                    all_labels[prefix].append(label)

        elif response.status_code == 401:
            print(f"  Warning: Snek API authentication failed (401)")
        else:
            print(f"  Warning: Snek API returned {response.status_code}: {response.text[:100]}")

    except requests.RequestException as e:
        print(f"  Warning: Snek API error: {e}")

    # Log all ontology mappings found
    if all_labels:
        print(f"  Snek found ontology mappings:")
        for prefix, ids in sorted(all_labels.items()):
            print(f"    {prefix}: {', '.join(ids)}")

    # Deduplicate by mesh_id
    seen_ids = set()
    unique_terms = []
    for term in mesh_terms:
        if term["mesh_id"] not in seen_ids:
            seen_ids.add(term["mesh_id"])
            unique_terms.append(term)

    # Cache the results
    _mesh_cache[cache_key] = unique_terms

    return unique_terms


def get_mesh_terms_from_snowflake(condition: str) -> list[dict]:
    """Fallback: Find MeSH terms directly from Snowflake by text search.

    Args:
        condition: Clinical condition name

    Returns:
        List of dicts with 'mesh_id', 'mesh_name'
    """
    service = get_snowflake_service()
    safe_condition = escape_sql_string(condition.lower())

    query = f"""
    SELECT DISTINCT
        DESCRIPTOR_UI as mesh_id,
        DESCRIPTOR_NAME as mesh_name,
        COUNT(DISTINCT PMID) as article_count
    FROM REFERENCE_PROD.SILVER_PUBMED.MESH_HEADINGS
    WHERE LOWER(DESCRIPTOR_NAME) LIKE '%{safe_condition}%'
    GROUP BY DESCRIPTOR_UI, DESCRIPTOR_NAME
    ORDER BY article_count DESC
    LIMIT 10
    """

    df = service.query_to_df(query)

    return [
        {
            "mesh_id": row["mesh_id"],
            "mesh_name": row["mesh_name"],
            "source": "snowflake_text_search",
            "article_count": row["article_count"],
        }
        for _, row in df.iterrows()
    ]


def expand_mesh_terms(condition: str, api_key: str = None) -> list[dict]:
    """Get MeSH terms for a condition using snek + fallback.

    Args:
        condition: Clinical condition name
        api_key: Optional snek API key

    Returns:
        List of MeSH term dicts
    """
    print(f"Expanding '{condition}' to MeSH terms via snek...")

    # Try snek first
    mesh_terms = get_mesh_terms_from_snek(condition, api_key)

    if mesh_terms:
        print(f"  Found {len(mesh_terms)} MeSH terms via snek")
        for term in mesh_terms:
            print(f"    - {term['mesh_name']} (MESH:{term['mesh_id']})")
    else:
        # Fallback to Snowflake text search
        print("  Snek returned no MeSH terms, falling back to Snowflake text search...")
        mesh_terms = get_mesh_terms_from_snowflake(condition)
        print(f"  Found {len(mesh_terms)} MeSH terms via Snowflake")
        for term in mesh_terms:
            print(f"    - {term['mesh_name']} (MESH:{term['mesh_id']})")

    return mesh_terms


def build_mesh_search_query(
    mesh_ids: list[str],
    ai_terms: list[str],
    max_results: int = 500,
    min_year: int = 2010,
) -> str:
    """Build SQL query to search PubMed by MeSH terms + AI keywords.

    Args:
        mesh_ids: List of MeSH descriptor UIDs (e.g., ['D003930'])
        ai_terms: List of AI-related keywords
        max_results: Maximum results to return
        min_year: Minimum publication year

    Returns:
        SQL query string
    """
    # Build MeSH ID conditions
    mesh_id_list = ", ".join(f"'{mid}'" for mid in mesh_ids)

    # Build AI term conditions (still search in title/abstract)
    ai_conditions = []
    for term in ai_terms:
        safe_term = escape_sql_string(term.lower())
        ai_conditions.append(
            f"(LOWER(a.TITLE) LIKE '%{safe_term}%' OR LOWER(a.ABSTRACT) LIKE '%{safe_term}%')"
        )
    ai_clause = " OR ".join(ai_conditions)

    return f"""
SELECT DISTINCT
    a.PMID,
    a.TITLE as title,
    a.ABSTRACT as abstract,
    a.PUB_YEAR as year,
    a.JOURNAL as journal,
    m.DESCRIPTOR_NAME as primary_mesh_term,
    m.DESCRIPTOR_UI as primary_mesh_id
FROM REFERENCE_PROD.SILVER_PUBMED.ARTICLES a
INNER JOIN REFERENCE_PROD.SILVER_PUBMED.MESH_HEADINGS m
    ON a.PMID = m.PMID
WHERE m.DESCRIPTOR_UI IN ({mesh_id_list})
  AND ({ai_clause})
  AND a.PUB_YEAR >= {min_year}
ORDER BY a.PUB_YEAR DESC
LIMIT {max_results}
"""


def search_pubmed_by_mesh(
    condition: str,
    max_results: int = 500,
    min_year: int = 2010,
    snek_api_key: str = None,
) -> list[dict]:
    """Search PubMed using MeSH terms from snek ontology mapping.

    Args:
        condition: Clinical condition to search
        max_results: Maximum results to return
        min_year: Minimum publication year
        snek_api_key: Optional snek API Bearer token

    Returns:
        List of article dictionaries
    """
    print("=" * 60)
    print("MeSH-based PubMed Search (via snek ontology)")
    print("=" * 60)

    # Step 1: Get MeSH terms via snek
    mesh_terms = expand_mesh_terms(condition, snek_api_key)

    if not mesh_terms:
        print("No MeSH terms found. Cannot proceed with MeSH-based search.")
        return []

    # Extract MeSH IDs
    mesh_ids = [t["mesh_id"] for t in mesh_terms]

    # Step 2: Build and execute query
    print(f"\nSearching PubMed with {len(mesh_ids)} MeSH terms + AI keywords...")

    service = get_snowflake_service()
    query = build_mesh_search_query(mesh_ids, AI_TERMS, max_results, min_year)

    df = service.query_to_df(query)
    print(f"Found {len(df)} articles")

    # Step 3: Fetch authors
    if len(df) > 0:
        print("Fetching authors...")
        from .snowflake_search import fetch_authors_for_pmids
        pmids = df['pmid'].astype(str).tolist()
        authors_by_pmid = fetch_authors_for_pmids(service, pmids)
    else:
        authors_by_pmid = {}

    # Step 4: Build article list
    articles = []
    for _, row in df.iterrows():
        pmid = str(row['pmid'])
        articles.append({
            "pmid": pmid,
            "title": row.get('title') or '',
            "abstract": row.get('abstract') or '',
            "journal": row.get('journal') or '',
            "year": str(row.get('year') or ''),
            "authors": authors_by_pmid.get(pmid, []),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "primary_mesh_term": row.get('primary_mesh_term') or '',
            "primary_mesh_id": row.get('primary_mesh_id') or '',
            "search_method": "mesh",
        })

    print(f"Retrieved {len(articles)} articles with metadata")
    return articles


# Convenience alias matching existing API
def search_clinical_ai_literature_mesh(
    condition: str,
    max_results: int = 500,
    snek_api_key: str = None,
) -> list[dict]:
    """Search PubMed for AI/ML literature using MeSH-based approach.

    Drop-in replacement for search_clinical_ai_literature that uses
    MeSH ontology mapping instead of text-based term expansion.

    Args:
        condition: Clinical condition to search
        max_results: Maximum results to return
        snek_api_key: Optional snek API Bearer token

    Returns:
        List of article dictionaries with MeSH metadata
    """
    return search_pubmed_by_mesh(condition, max_results=max_results, snek_api_key=snek_api_key)
