"""Problem framework classification for research gap analysis."""

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_client import get_completion

# Model for classification - Haiku for speed
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "haiku")

# Default problem frameworks for clinical AI/ML research
DEFAULT_PROBLEM_FRAMEWORKS = {
    "screening_detection": {
        "name": "Screening & Detection",
        "description": "Automated screening, early detection, disease identification",
        "keywords": ["screening", "detection", "identify", "diagnose", "diagnosis"]
    },
    "grading_staging": {
        "name": "Grading & Staging",
        "description": "Severity grading, disease staging, classification of disease levels",
        "keywords": ["grading", "staging", "severity", "classification", "mild", "moderate", "severe"]
    },
    "progression_prediction": {
        "name": "Progression Prediction",
        "description": "Predicting disease progression, time to progression, risk stratification",
        "keywords": ["progression", "predict", "prognosis", "risk", "future", "longitudinal"]
    },
    "treatment_response": {
        "name": "Treatment Response",
        "description": "Predicting treatment outcomes, response to therapy, personalized treatment",
        "keywords": ["treatment", "therapy", "response", "outcome", "anti-VEGF", "intervention"]
    },
    "biomarker_discovery": {
        "name": "Biomarker Discovery",
        "description": "Identifying new biomarkers, imaging biomarkers, predictive features",
        "keywords": ["biomarker", "feature", "indicator", "marker", "signature"]
    },
    "segmentation_quantification": {
        "name": "Segmentation & Quantification",
        "description": "Lesion segmentation, anatomical segmentation, quantitative measurements",
        "keywords": ["segmentation", "segment", "quantification", "measurement", "area", "volume"]
    },
    "explainability_interpretability": {
        "name": "Explainability & Interpretability",
        "description": "Model interpretability, explainable AI, attention maps, feature visualization",
        "keywords": ["explainability", "interpretability", "attention", "heatmap", "visualization", "explain"]
    },
    "clinical_validation": {
        "name": "Clinical Validation",
        "description": "Real-world validation, clinical trials, deployment studies",
        "keywords": ["validation", "clinical", "real-world", "deployment", "implementation", "trial"]
    },
    "multimodal_fusion": {
        "name": "Multimodal Fusion",
        "description": "Combining multiple imaging modalities, data fusion",
        "keywords": ["multimodal", "fusion", "combine", "OCT", "fundus", "multiple"]
    },
    "dataset_benchmark": {
        "name": "Dataset & Benchmark",
        "description": "New datasets, benchmarking studies, model comparison",
        "keywords": ["dataset", "benchmark", "comparison", "evaluation", "public"]
    }
}


CLASSIFICATION_SYSTEM_PROMPT = """You are an expert at classifying scientific papers into research problem categories.

Given a paper's title and abstract, classify it into ONE OR MORE of the provided problem categories.
Only select categories that are clearly addressed by the paper.

Return your response as a JSON object:
{
    "pmid": "the paper's PMID",
    "categories": ["category_id_1", "category_id_2"],
    "primary_category": "main_category_id",
    "confidence": 0.0-1.0
}

IMPORTANT: Only return the JSON object, no other text."""


def classify_papers_by_problem(
    articles: list[dict],
    clinical_term: str,
    problem_frameworks: dict = None,
    batch_size: int = 10,
    max_workers: int = 10,
) -> tuple[list[dict], dict]:
    """Classify papers by research problem framework.

    Args:
        articles: List of article dicts with 'pmid', 'title', 'abstract'
        clinical_term: The clinical condition being studied
        problem_frameworks: Dict of problem categories (uses defaults if None)
        batch_size: Number of articles per LLM call
        max_workers: Number of parallel workers

    Returns:
        Tuple of (classified_articles, category_counts)
    """
    if not articles:
        return [], {}

    frameworks = problem_frameworks or DEFAULT_PROBLEM_FRAMEWORKS

    print(f"\nClassifying {len(articles)} articles by research problem...")
    print(f"Problem categories: {len(frameworks)}")
    print(f"Model: {CLASSIFICATION_MODEL}, batch size: {batch_size}, parallel workers: {max_workers}")

    # Batch articles
    batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]

    # Classify each batch in parallel
    all_classifications = {}
    completed = 0

    def classify_batch(batch):
        return _classify_batch(batch, clinical_term, frameworks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            completed += 1
            batch_results = future.result()
            all_classifications.update(batch_results)
            if completed % 5 == 0 or completed == len(batches):
                print(f"  Progress: {completed}/{len(batches)} batches ({len(all_classifications)} articles classified)")

    # Apply classifications to articles and count categories
    category_counts = Counter()
    classified_articles = []

    for article in articles:
        pmid = str(article['pmid'])
        if pmid in all_classifications:
            classification = all_classifications[pmid]
            article_copy = article.copy()
            article_copy['problem_categories'] = classification.get('categories', [])
            article_copy['primary_problem'] = classification.get('primary_category', '')
            article_copy['classification_confidence'] = classification.get('confidence', 0)
            classified_articles.append(article_copy)

            # Count each category
            for cat in classification.get('categories', []):
                category_counts[cat] += 1
        else:
            classified_articles.append(article)

    # Build summary with category names
    category_summary = {}
    for cat_id, count in category_counts.items():
        if cat_id in frameworks:
            category_summary[cat_id] = {
                'name': frameworks[cat_id]['name'],
                'count': count,
                'percentage': round(count / len(articles) * 100, 1)
            }

    print(f"\nClassification complete!")
    return classified_articles, category_summary


def _classify_batch(articles: list[dict], clinical_term: str, frameworks: dict) -> dict[str, dict]:
    """Classify a batch of articles."""
    # Build category descriptions for prompt
    category_desc = "\n".join([
        f"- {cat_id}: {info['name']} - {info['description']}"
        for cat_id, info in frameworks.items()
    ])

    # Build article summaries
    article_texts = []
    for i, article in enumerate(articles, 1):
        title = article.get('title', '')[:200]
        abstract = article.get('abstract', '')[:400]
        pmid = article.get('pmid', '')
        article_texts.append(f"""Article {i} (PMID: {pmid}):
Title: {title}
Abstract: {abstract}
""")

    prompt = f"""Clinical condition: {clinical_term}

Problem categories for classification:
{category_desc}

Classify each article below into the most relevant problem categories.
Each article can belong to multiple categories if applicable.

{chr(10).join(article_texts)}

Return a JSON array with classification for each article."""

    try:
        response = get_completion(prompt, CLASSIFICATION_SYSTEM_PROMPT, model=CLASSIFICATION_MODEL)

        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        results = json.loads(response)

        # Convert to dict by PMID
        classifications = {}
        if isinstance(results, list):
            for item in results:
                pmid = str(item.get('pmid', ''))
                classifications[pmid] = item
        elif isinstance(results, dict) and 'pmid' in results:
            # Single result
            classifications[str(results['pmid'])] = results

        return classifications

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Classification failed for batch: {e}")
        return {}


def generate_heatmap_data(category_summary: dict, frameworks: dict = None) -> dict:
    """Generate data for heatmap visualization.

    Returns dict with:
    - categories: list of category names
    - counts: list of counts
    - percentages: list of percentages
    - density_levels: list of density classifications (sparse/moderate/dense)
    """
    frameworks = frameworks or DEFAULT_PROBLEM_FRAMEWORKS

    # Sort by count descending
    sorted_cats = sorted(category_summary.items(), key=lambda x: x[1]['count'], reverse=True)

    categories = []
    counts = []
    percentages = []
    density_levels = []

    max_count = max([c['count'] for c in category_summary.values()]) if category_summary else 1

    for cat_id, info in sorted_cats:
        categories.append(info['name'])
        counts.append(info['count'])
        percentages.append(info['percentage'])

        # Classify density
        ratio = info['count'] / max_count
        if ratio >= 0.6:
            density_levels.append('dense')
        elif ratio >= 0.3:
            density_levels.append('moderate')
        else:
            density_levels.append('sparse')

    return {
        'categories': categories,
        'counts': counts,
        'percentages': percentages,
        'density_levels': density_levels,
        'total_articles': sum(counts)
    }


def print_heatmap_ascii(heatmap_data: dict):
    """Print ASCII heatmap to terminal."""
    print("\n" + "=" * 70)
    print("RESEARCH PROBLEM HEATMAP - Literature Density Analysis")
    print("=" * 70)

    max_count = max(heatmap_data['counts']) if heatmap_data['counts'] else 1
    bar_width = 40

    for i, cat in enumerate(heatmap_data['categories']):
        count = heatmap_data['counts'][i]
        pct = heatmap_data['percentages'][i]
        density = heatmap_data['density_levels'][i]

        # Create bar
        bar_len = int((count / max_count) * bar_width)
        if density == 'dense':
            bar = '█' * bar_len
            indicator = '[DENSE]  '
        elif density == 'moderate':
            bar = '▓' * bar_len
            indicator = '[MODERATE]'
        else:
            bar = '░' * bar_len
            indicator = '[SPARSE] '

        # Pad bar
        bar = bar.ljust(bar_width)

        print(f"\n{cat[:30]:<30}")
        print(f"  {bar} {count:>4} ({pct:>5.1f}%) {indicator}")

    print("\n" + "-" * 70)
    print("Legend: █ Dense (>60%)  ▓ Moderate (30-60%)  ░ Sparse (<30%)")
    print(f"Total classifications: {heatmap_data['total_articles']}")
    print("(Papers can belong to multiple categories)")
    print("=" * 70)


def identify_research_gaps(category_summary: dict, frameworks: dict = None) -> list[dict]:
    """Identify potential research gaps based on sparse categories."""
    frameworks = frameworks or DEFAULT_PROBLEM_FRAMEWORKS

    if not category_summary:
        return []

    max_count = max([c['count'] for c in category_summary.values()])

    gaps = []
    for cat_id, info in category_summary.items():
        ratio = info['count'] / max_count
        if ratio < 0.3:  # Sparse
            gaps.append({
                'category_id': cat_id,
                'category_name': info['name'],
                'count': info['count'],
                'percentage': info['percentage'],
                'gap_severity': 'high' if ratio < 0.15 else 'moderate',
                'description': frameworks[cat_id]['description'] if cat_id in frameworks else ''
            })

    # Sort by count ascending (most sparse first)
    gaps.sort(key=lambda x: x['count'])
    return gaps
