"""PubMed search via Snowflake (REFERENCE_PROD.SILVER_PUBMED)."""

import os
import pandas as pd
import snowflake.connector
from typing import Optional
from .config import load_config, get_secret

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


AI_TERMS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
]


# Global singleton for connection reuse
_snowflake_service_instance = None


class SnowflakeSearchService:
    """Service for searching PubMed via Snowflake.

    Uses singleton pattern to reuse connection across calls within a session.
    Supports both browser SSO (local) and password auth (Streamlit Cloud).
    """

    def __init__(self):
        self._connection = None
        self._config = {
            "account": get_secret("SNOWFLAKE_ACCOUNT"),
            "user": get_secret("SNOWFLAKE_USER"),
            "warehouse": get_secret("SNOWFLAKE_WAREHOUSE"),
            "database": get_secret("SNOWFLAKE_DATABASE", "REFERENCE_PROD"),
            "schema": get_secret("SNOWFLAKE_SCHEMA", "SILVER_PUBMED"),
            "role": get_secret("SNOWFLAKE_ROLE"),
            "client_session_keep_alive": True,
        }

        # Use password auth if provided (for Streamlit Cloud), otherwise use browser SSO
        password = get_secret("SNOWFLAKE_PASSWORD")
        if password:
            self._config["password"] = password
        else:
            self._config["authenticator"] = "externalbrowser"
            self._config["client_store_temporary_credential"] = True

    def _get_connection(self):
        """Get or create Snowflake connection."""
        if self._connection is None or self._connection.is_closed():
            print("Connecting to Snowflake (browser auth - one-time per session)...")
            self._connection = snowflake.connector.connect(**self._config)
            print("Connected to Snowflake!")
        return self._connection

    def query_to_df(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return DataFrame."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
            df = cur.fetch_pandas_all()
            df.columns = [c.lower() for c in df.columns]
            return df

    def close(self):
        """Close connection (usually not needed - connection is kept alive)."""
        if self._connection and not self._connection.is_closed():
            self._connection.close()
            self._connection = None


def get_snowflake_service() -> SnowflakeSearchService:
    """Get or create the singleton SnowflakeSearchService instance.

    This ensures only one browser auth prompt per Python session.
    """
    global _snowflake_service_instance
    if _snowflake_service_instance is None:
        _snowflake_service_instance = SnowflakeSearchService()
    return _snowflake_service_instance


def escape_sql_string(s: str) -> str:
    """Escape single quotes for SQL."""
    return s.replace("'", "''")


def build_search_query(clinical_terms: list[str], ai_terms: list[str], max_results: int = 500) -> str:
    """Build SQL query to search PubMed articles in Snowflake.

    Searches for articles where:
    - Title OR Abstract contains any of the clinical terms
    - AND Title OR Abstract contains any of the AI terms

    Uses word boundary matching for short terms (<=3 chars) to avoid false positives.
    """
    # Build clinical term conditions
    clinical_conditions = []
    for term in clinical_terms:
        safe_term = escape_sql_string(term.lower())
        # For short terms (like "DR"), use word boundary matching
        if len(term) <= 3:
            # Match as standalone word using regex
            clinical_conditions.append(
                f"(REGEXP_LIKE(LOWER(a.TITLE), '\\\\b{safe_term}\\\\b') OR REGEXP_LIKE(LOWER(a.ABSTRACT), '\\\\b{safe_term}\\\\b'))"
            )
        else:
            clinical_conditions.append(
                f"(LOWER(a.TITLE) LIKE '%{safe_term}%' OR LOWER(a.ABSTRACT) LIKE '%{safe_term}%')"
            )
    clinical_clause = " OR ".join(clinical_conditions)

    # Build AI term conditions
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
    a.JOURNAL as journal
FROM REFERENCE_PROD.SILVER_PUBMED.ARTICLES a
WHERE ({clinical_clause})
  AND ({ai_clause})
  AND a.PUB_YEAR >= 2010
ORDER BY a.PUB_YEAR DESC
LIMIT {max_results}
"""


def fetch_authors_for_pmids(service: SnowflakeSearchService, pmids: list[str]) -> dict[str, list[str]]:
    """Fetch authors for a list of PMIDs."""
    if not pmids:
        return {}

    # Batch into chunks of 1000 to avoid query limits
    authors_by_pmid = {}
    batch_size = 1000

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        pmid_list = ",".join(f"'{p}'" for p in batch)

        query = f"""
SELECT
    PMID,
    FORE_NAME,
    LAST_NAME
FROM REFERENCE_PROD.SILVER_PUBMED.AUTHORS
WHERE PMID IN ({pmid_list})
ORDER BY PMID, AUTHOR_ORDER
"""

        df = service.query_to_df(query)

        # Group authors by PMID
        for _, row in df.iterrows():
            pmid = str(row['pmid'])
            fore_name = row.get('fore_name', '') or ''
            last_name = row.get('last_name', '') or ''
            name = f"{fore_name} {last_name}".strip()
            if pmid not in authors_by_pmid:
                authors_by_pmid[pmid] = []
            if name:
                authors_by_pmid[pmid].append(name)

    return authors_by_pmid


def search_clinical_ai_literature_snowflake(
    clinical_terms: list[str],
    max_results: int = 500,
) -> list[dict]:
    """Search PubMed via Snowflake for AI/ML literature on clinical terms.

    Args:
        clinical_terms: List of clinical terms (including synonyms)
        max_results: Maximum total results to return

    Returns:
        List of article dictionaries
    """
    # Use singleton to reuse connection (only one browser auth per session)
    service = get_snowflake_service()

    print(f"Searching Snowflake PubMed with {len(clinical_terms)} clinical terms x {len(AI_TERMS)} AI terms...")

    # Build and execute single query
    query = build_search_query(clinical_terms, AI_TERMS, max_results)
    print("Executing query...")
    df = service.query_to_df(query)

    print(f"Found {len(df)} articles. Fetching authors...")

    # Fetch authors
    pmids = df['pmid'].astype(str).tolist()
    authors_by_pmid = fetch_authors_for_pmids(service, pmids)

    # Build article list
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
        })

    print(f"Retrieved {len(articles)} articles with metadata.")
    return articles
    # Note: Connection is kept alive for reuse, not closed


# Convenience function matching the old API
def search_clinical_ai_literature(
    clinical_terms: list[str],
    max_results_per_query: int = 50,
    max_workers: int = 3,
) -> list[dict]:
    """Search PubMed via Snowflake (drop-in replacement for NCBI API version).

    Note: max_results_per_query and max_workers are ignored - Snowflake
    does everything in a single fast query.
    """
    # Convert to total max results (roughly equivalent)
    max_results = max_results_per_query * len(clinical_terms) * len(AI_TERMS)
    max_results = min(max_results, 1000)  # Cap at 1000

    return search_clinical_ai_literature_snowflake(clinical_terms, max_results)
