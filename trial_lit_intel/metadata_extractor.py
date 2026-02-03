"""Extract additional metadata from paper abstracts using LLM."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .llm_client import get_completion
except ImportError:
    from trial_lit_intel.llm_client import get_completion

# Model for metadata extraction - Haiku for speed
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "haiku")

EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting research metadata from scientific paper abstracts.

For each paper, extract:
1. external_validation: Did the study validate on an external/independent dataset? (yes/no/unclear)
2. public_code: Is code/model publicly available? (yes/no/unclear)
3. sample_size: Total sample size (number of patients/images/samples). Return the number or "not reported"

Return your response as a JSON array with objects containing:
- "pmid": the article PMID
- "external_validation": "yes", "no", or "unclear"
- "public_code": "yes", "no", or "unclear"
- "sample_size": number or "not reported"

IMPORTANT: Only return the JSON array, no other text."""


def extract_paper_metadata(
    articles: list[dict],
    batch_size: int = 10,
    max_workers: int = 10,
) -> list[dict]:
    """Extract additional metadata from papers using LLM.

    Args:
        articles: List of article dicts with 'pmid', 'title', 'abstract'
        batch_size: Number of articles per LLM call
        max_workers: Number of parallel workers

    Returns:
        List of articles with added metadata fields
    """
    if not articles:
        return []

    print(f"\nExtracting metadata from {len(articles)} articles...")
    print(f"Model: {EXTRACTION_MODEL}, batch size: {batch_size}, parallel workers: {max_workers}")

    # Batch articles
    batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]

    # Extract from each batch in parallel
    all_metadata = {}
    completed = 0

    def extract_batch(batch):
        return _extract_batch(batch)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            completed += 1
            batch_results = future.result()
            all_metadata.update(batch_results)
            if completed % 5 == 0 or completed == len(batches):
                print(f"  Progress: {completed}/{len(batches)} batches ({len(all_metadata)} articles processed)")

    # Apply metadata to articles
    enriched_articles = []
    for article in articles:
        pmid = str(article['pmid'])
        article_copy = article.copy()

        if pmid in all_metadata:
            meta = all_metadata[pmid]
            article_copy['external_validation'] = meta.get('external_validation', 'unclear')
            article_copy['public_code'] = meta.get('public_code', 'unclear')
            article_copy['sample_size'] = meta.get('sample_size', 'not reported')
        else:
            article_copy['external_validation'] = 'unclear'
            article_copy['public_code'] = 'unclear'
            article_copy['sample_size'] = 'not reported'

        enriched_articles.append(article_copy)

    print(f"Metadata extraction complete!")
    return enriched_articles


def _extract_batch(articles: list[dict]) -> dict[str, dict]:
    """Extract metadata from a batch of articles."""
    # Build article summaries
    article_texts = []
    for i, article in enumerate(articles, 1):
        title = article.get('title', '')[:200]
        abstract = article.get('abstract', '')[:600]
        pmid = article.get('pmid', '')
        article_texts.append(f"""Article {i} (PMID: {pmid}):
Title: {title}
Abstract: {abstract}
""")

    prompt = f"""Extract metadata from each article below.

{chr(10).join(article_texts)}

Return a JSON array with metadata for each article."""

    try:
        response = get_completion(prompt, EXTRACTION_SYSTEM_PROMPT, model=EXTRACTION_MODEL)

        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        results = json.loads(response)

        # Convert to dict by PMID
        metadata = {}
        if isinstance(results, list):
            for item in results:
                pmid = str(item.get('pmid', ''))
                metadata[pmid] = item
        elif isinstance(results, dict) and 'pmid' in results:
            metadata[str(results['pmid'])] = results

        return metadata

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Metadata extraction failed for batch: {e}")
        return {}
