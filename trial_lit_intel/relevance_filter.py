"""LLM-based relevance filtering for articles."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_client import get_completion


# Model for relevance scoring - Haiku is fast and cheap
# Can be overridden via env var RELEVANCE_SCORING_MODEL
SCORING_MODEL = os.getenv("RELEVANCE_SCORING_MODEL", "haiku")

RELEVANCE_SYSTEM_PROMPT = """You are an expert at evaluating scientific literature relevance.

Your task is to score how relevant each article is to the application of AI/ML/deep learning
to a specific clinical condition.

Score each article from 0-10:
- 10: Directly about applying deep learning/AI/ML to diagnose, predict, or treat the condition
- 7-9: Strongly related - AI/ML applied to the condition or closely related aspects
- 4-6: Moderately related - mentions both AI/ML and the condition but not as main focus
- 1-3: Weakly related - tangentially mentions one or both topics
- 0: Not related - about something else entirely

Return your response as a JSON array with objects containing:
- "pmid": the article PMID
- "score": relevance score (0-10)
- "reason": brief explanation (1 sentence)

IMPORTANT: Only return the JSON array, no other text."""


def score_article_relevance(
    articles: list[dict],
    clinical_term: str,
    batch_size: int = 25,  # OPTIMIZED: Larger batches = fewer API calls
    min_score: int = 7,
    max_workers: int = 15,  # OPTIMIZED: More parallel workers
) -> list[dict]:
    """Score articles for relevance using LLM and filter by minimum score.

    Args:
        articles: List of article dicts with 'pmid', 'title', 'abstract'
        clinical_term: The clinical condition being searched
        batch_size: Number of articles to score per LLM call
        min_score: Minimum relevance score to include (0-10)
        max_workers: Number of parallel LLM calls

    Returns:
        List of articles with score >= min_score, sorted by score descending
    """
    if not articles:
        return []

    model_name = SCORING_MODEL or "default (opus)"
    print(f"Scoring {len(articles)} articles for relevance to '{clinical_term}'...")
    print(f"Model: {model_name}, batch size: {batch_size}, parallel workers: {max_workers}")

    # Batch articles
    batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]

    # Score each batch in parallel
    all_scores = {}
    completed = 0

    def score_batch(batch):
        return _score_batch(batch, clinical_term)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            completed += 1
            batch_scores = future.result()
            all_scores.update(batch_scores)
            # Show progress less frequently for cleaner output
            if completed % 5 == 0 or completed == len(batches):
                print(f"  Progress: {completed}/{len(batches)} batches ({len(all_scores)} articles scored)")

    # Filter and sort by score
    scored_articles = []
    for article in articles:
        pmid = article['pmid']
        if pmid in all_scores:
            score_info = all_scores[pmid]
            article_with_score = article.copy()
            article_with_score['relevance_score'] = score_info['score']
            article_with_score['relevance_reason'] = score_info.get('reason', '')

            if score_info['score'] >= min_score:
                scored_articles.append(article_with_score)

    # Sort by relevance score descending
    scored_articles.sort(key=lambda x: x['relevance_score'], reverse=True)

    print(f"Found {len(scored_articles)} articles with relevance score >= {min_score}")
    return scored_articles


def _score_batch(articles: list[dict], clinical_term: str) -> dict[str, dict]:
    """Score a batch of articles using LLM (Haiku for speed).

    Returns:
        Dict mapping PMID to {score, reason}
    """
    # Build prompt with article summaries
    article_texts = []
    for i, article in enumerate(articles, 1):
        title = article.get('title', '')[:200]
        abstract = article.get('abstract', '')[:500]  # Truncate to manage tokens
        pmid = article.get('pmid', '')

        article_texts.append(f"""Article {i} (PMID: {pmid}):
Title: {title}
Abstract: {abstract}
""")

    prompt = f"""Clinical condition: {clinical_term}

Score the relevance of each article below to the application of AI/ML/deep learning
for diagnosing, predicting outcomes, or treating "{clinical_term}".

{chr(10).join(article_texts)}

Return a JSON array with scores for each article."""

    try:
        response = get_completion(prompt, RELEVANCE_SYSTEM_PROMPT, model=SCORING_MODEL)

        # Parse JSON from response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        scores = json.loads(response)

        # Convert to dict
        result = {}
        for item in scores:
            pmid = str(item.get('pmid', ''))
            result[pmid] = {
                'score': int(item.get('score', 0)),
                'reason': item.get('reason', '')
            }
        return result

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Failed to parse LLM response: {e}")
        # Return neutral scores on failure
        return {str(a['pmid']): {'score': 5, 'reason': 'Scoring failed'} for a in articles}


def quick_relevance_check(title: str, abstract: str, clinical_term: str) -> tuple[int, str]:
    """Quick single-article relevance check using Haiku.

    Returns:
        Tuple of (score, reason)
    """
    prompt = f"""Clinical condition: {clinical_term}

Score this article's relevance (0-10) to applying AI/ML/deep learning for "{clinical_term}":

Title: {title}
Abstract: {abstract[:800]}

Return JSON: {{"score": N, "reason": "brief explanation"}}"""

    try:
        response = get_completion(prompt, RELEVANCE_SYSTEM_PROMPT, model=SCORING_MODEL)
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        data = json.loads(response)
        return int(data.get('score', 0)), data.get('reason', '')
    except:
        return 5, "Scoring failed"
