"""NPDR AI Literature Tracker - Main Pipeline.

Orchestrates the tiered extraction system:
1. Abstract extraction
2. PMC full text (if available)
3. bioRxiv/medRxiv search (if not in PMC)
4. Manual review flag (if no full text found)
"""

import csv
import os
from datetime import datetime
from typing import Optional

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .pmc_fetcher import batch_fetch_pmc_full_text, check_pmc_availability
from .biorxiv_fetcher import batch_search_biorxiv, fetch_biorxiv_full_text
from .extractor import batch_extract, NPDRExtractor


# CSV column mapping (matches NPDR_AI_Literature_Tracker.csv)
CSV_COLUMNS = [
    "Paper Title",
    "First Author",
    "Year",
    "Journal/Conference",
    "DOI/URL",
    "Task Type",
    "Imaging Modality",
    "Other Input Features",
    "Prediction Target",
    "Prediction Horizon",
    "Model Architecture",
    "Training Dataset",
    "Dataset Size (n)",
    "External Validation?",
    "Primary Metric",
    "Primary Metric Value",
    "Secondary Metrics",
    "Addresses Long-term Prediction (Ch1)",
    "Identifies Early Signals (Ch2)",
    "Handles Low Event Rates (Ch3)",
    "Identifies Rapid Progressors (Ch4)",
    "Improves Grading Consistency (Ch5)",
    "Relevance to BLKR-201 Study",
    "Potential Application",
    "Key Findings/Notes",
    "Limitations",
    "Data Source",
    "Extraction Confidence",
    "Needs Manual Review",
]

# Field mapping from extracted data to CSV columns
FIELD_TO_COLUMN = {
    "title": "Paper Title",
    "first_author": "First Author",
    "year": "Year",
    "journal": "Journal/Conference",
    "url": "DOI/URL",
    "task_type": "Task Type",
    "imaging_modality": "Imaging Modality",
    "other_input_features": "Other Input Features",
    "prediction_target": "Prediction Target",
    "prediction_horizon": "Prediction Horizon",
    "model_architecture": "Model Architecture",
    "training_dataset": "Training Dataset",
    "dataset_size": "Dataset Size (n)",
    "external_validation": "External Validation?",
    "primary_metric": "Primary Metric",
    "primary_metric_value": "Primary Metric Value",
    "secondary_metrics": "Secondary Metrics",
    "addresses_long_term_prediction_ch1": "Addresses Long-term Prediction (Ch1)",
    "identifies_early_signals_ch2": "Identifies Early Signals (Ch2)",
    "handles_low_event_rates_ch3": "Handles Low Event Rates (Ch3)",
    "identifies_rapid_progressors_ch4": "Identifies Rapid Progressors (Ch4)",
    "improves_grading_consistency_ch5": "Improves Grading Consistency (Ch5)",
    "relevance_to_blkr201": "Relevance to BLKR-201 Study",
    "potential_application": "Potential Application",
    "key_findings": "Key Findings/Notes",
    "limitations": "Limitations",
    "data_source": "Data Source",
    "extraction_confidence": "Extraction Confidence",
    "needs_manual_review": "Needs Manual Review",
}


class NPDRTracker:
    """NPDR AI Literature Tracker Pipeline."""

    def __init__(self, ncbi_api_key: str = None):
        """Initialize tracker.

        Args:
            ncbi_api_key: Optional NCBI API key for higher rate limits
        """
        self.ncbi_api_key = ncbi_api_key or os.getenv("NCBI_API_KEY")
        self.extractor = NPDRExtractor()

    def process_papers(
        self,
        papers: list,
        skip_full_text: bool = False,
        max_workers: int = 5,
    ) -> list:
        """Process papers through the tiered extraction pipeline.

        Args:
            papers: List of paper dicts from PubMed search
            skip_full_text: If True, only use abstracts (faster)
            max_workers: Number of parallel workers

        Returns:
            List of papers with extracted fields
        """
        print("=" * 70)
        print("NPDR AI Literature Tracker - Tiered Extraction Pipeline")
        print("=" * 70)
        print(f"Processing {len(papers)} papers")

        # Prepare papers with first author
        for paper in papers:
            authors = paper.get("authors", [])
            paper["first_author"] = authors[0] if authors else "Unknown"

        if skip_full_text:
            print("\nSkipping full text retrieval (abstract-only mode)")
            full_texts = {}
        else:
            # Tier 2: Check PMC availability and fetch full text
            full_texts = self._fetch_full_texts(papers, max_workers)

        # Extract structured data
        print("\n" + "-" * 70)
        print("EXTRACTION PHASE")
        print("-" * 70)

        extracted_papers = batch_extract(
            papers,
            full_texts=full_texts,
            max_workers=max_workers,
        )

        # Mark papers needing manual review
        for paper in extracted_papers:
            source = paper.get("data_source", "")
            if "Abstract" in source and "Full Text" not in source:
                # Check if critical fields are missing
                critical_missing = any(
                    paper.get(field, "Not reported") in ["Not reported", "N/A", "Extraction failed"]
                    for field in ["dataset_size", "model_architecture", "external_validation"]
                )
                paper["needs_manual_review"] = "Yes" if critical_missing else "No"
            else:
                paper["needs_manual_review"] = "No"

        # Summary
        self._print_summary(extracted_papers)

        return extracted_papers

    def _fetch_full_texts(self, papers: list, max_workers: int) -> dict:
        """Fetch full texts from PMC and bioRxiv.

        Returns:
            Dict mapping PMID to full text info
        """
        full_texts = {}
        pmids = [str(p.get("pmid", "")) for p in papers]

        # Tier 2: Check PMC availability
        print("\n" + "-" * 70)
        print("TIER 2: PubMed Central Full Text")
        print("-" * 70)

        pmc_results = batch_fetch_pmc_full_text(
            pmids,
            api_key=self.ncbi_api_key,
            max_workers=max_workers,
        )

        for pmid, result in pmc_results.items():
            if result:
                full_texts[pmid] = result

        # Tier 3: Search bioRxiv for papers not in PMC
        papers_without_fulltext = [
            p for p in papers
            if str(p.get("pmid", "")) not in full_texts
        ]

        if papers_without_fulltext:
            print("\n" + "-" * 70)
            print("TIER 3: bioRxiv/medRxiv Search")
            print("-" * 70)

            biorxiv_results = batch_search_biorxiv(
                papers_without_fulltext,
                max_workers=3,  # Conservative for bioRxiv
            )

            # Fetch full text for found preprints
            for pmid, result in biorxiv_results.items():
                if result and result.get("found"):
                    # bioRxiv API only gives abstract, note this
                    full_texts[pmid] = {
                        "pmid": pmid,
                        "full_text": result.get("abstract", ""),
                        "source": f"bioRxiv ({result.get('server', 'biorxiv')})",
                        "is_preprint": True,
                        "note": result.get("note", "Data from preprint; may differ from published version"),
                        "preprint_url": result.get("url", ""),
                        "match_score": result.get("match_score", 0),
                    }

        return full_texts

    def _print_summary(self, papers: list):
        """Print extraction summary."""
        print("\n" + "=" * 70)
        print("EXTRACTION SUMMARY")
        print("=" * 70)

        # Count by data source
        sources = {}
        for paper in papers:
            source = paper.get("data_source", "Unknown")
            sources[source] = sources.get(source, 0) + 1

        print("\nData sources:")
        for source, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {source}: {count} papers")

        # Count needing manual review
        needs_review = sum(1 for p in papers if p.get("needs_manual_review") == "Yes")
        print(f"\nNeeds manual review: {needs_review}/{len(papers)} papers")

        # Challenge coverage
        print("\nChallenge coverage:")
        challenges = [
            ("Ch1 - Long-term Prediction", "addresses_long_term_prediction_ch1"),
            ("Ch2 - Early Signals", "identifies_early_signals_ch2"),
            ("Ch3 - Low Event Rates", "handles_low_event_rates_ch3"),
            ("Ch4 - Rapid Progressors", "identifies_rapid_progressors_ch4"),
            ("Ch5 - Grading Consistency", "improves_grading_consistency_ch5"),
        ]

        for name, field in challenges:
            yes_count = sum(1 for p in papers if p.get(field) == "Y")
            partial_count = sum(1 for p in papers if p.get(field) == "Partial")
            print(f"  {name}: {yes_count} Yes, {partial_count} Partial")

        # Relevance distribution
        print("\nRelevance to BLKR-201:")
        for level in ["High", "Medium", "Low"]:
            count = sum(1 for p in papers if p.get("relevance_to_blkr201") == level)
            print(f"  {level}: {count} papers")

    def export_to_csv(self, papers: list, output_path: str):
        """Export extracted papers to CSV in tracker format.

        Args:
            papers: List of extracted paper dicts
            output_path: Path to output CSV file
        """
        print(f"\nExporting to CSV: {output_path}")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            for paper in papers:
                row = {}
                for field, column in FIELD_TO_COLUMN.items():
                    value = paper.get(field, "")

                    # Handle special cases
                    if field == "url" and not value:
                        pmid = paper.get("pmid", "")
                        value = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

                    row[column] = value

                writer.writerow(row)

        print(f"Exported {len(papers)} papers to {output_path}")

    def run_full_pipeline(
        self,
        clinical_term: str = "diabetic retinopathy",
        max_results: int = 50,
        min_relevance: int = 7,
        output_path: str = None,
        skip_full_text: bool = False,
    ) -> list:
        """Run the complete NPDR pipeline from search to CSV export.

        Args:
            clinical_term: Clinical term to search (default: diabetic retinopathy)
            max_results: Max results per search query
            min_relevance: Minimum relevance score
            output_path: Path to save CSV (auto-generated if None)
            skip_full_text: Skip full text retrieval (faster)

        Returns:
            List of extracted papers
        """
        from trial_lit_intel.term_expander import expand_clinical_term
        from trial_lit_intel.snowflake_search import search_clinical_ai_literature
        from trial_lit_intel.relevance_filter import score_article_relevance
        from trial_lit_intel.journal_filter import filter_by_journal_quality

        print("=" * 70)
        print("NPDR AI Literature Tracker - Full Pipeline")
        print("=" * 70)
        print(f"Clinical term: {clinical_term}")
        print(f"Max results: {max_results}")
        print(f"Min relevance: {min_relevance}")
        print()

        # Step 1: Expand terms
        print("Step 1: Expanding clinical term...")
        terms = expand_clinical_term(clinical_term)
        print(f"  Expanded to {len(terms)} terms")

        # Step 2: Search PubMed
        print("\nStep 2: Searching PubMed via Snowflake...")
        articles = search_clinical_ai_literature(terms, max_results_per_query=max_results)
        print(f"  Found {len(articles)} articles")

        if not articles:
            print("No articles found!")
            return []

        # Step 3: Filter by journal quality
        print("\nStep 3: Filtering by journal quality...")
        articles = filter_by_journal_quality(articles, tier_threshold=2)
        print(f"  {len(articles)} articles from quality journals")

        # Step 4: Filter by relevance
        print("\nStep 4: Filtering by relevance...")
        articles = score_article_relevance(
            articles,
            clinical_term=clinical_term,
            min_score=min_relevance,
        )
        print(f"  {len(articles)} articles passed relevance filter")

        if not articles:
            print("No relevant articles found!")
            return []

        # Step 5: Tiered extraction
        print("\nStep 5: Tiered extraction...")
        extracted = self.process_papers(articles, skip_full_text=skip_full_text)

        # Step 6: Export to CSV
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"npdr_tracker_{timestamp}.csv"

        self.export_to_csv(extracted, output_path)

        return extracted


def main():
    """CLI entry point for NPDR tracker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NPDR AI Literature Tracker - Extract structured data from DR literature"
    )
    parser.add_argument(
        "--term",
        default="diabetic retinopathy",
        help="Clinical term to search (default: diabetic retinopathy)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Max results per search query (default: 50)",
    )
    parser.add_argument(
        "--min-relevance",
        type=int,
        default=7,
        help="Minimum relevance score (default: 7)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--skip-full-text",
        action="store_true",
        help="Skip full text retrieval (faster, less complete)",
    )

    args = parser.parse_args()

    tracker = NPDRTracker()
    tracker.run_full_pipeline(
        clinical_term=args.term,
        max_results=args.max_results,
        min_relevance=args.min_relevance,
        output_path=args.output,
        skip_full_text=args.skip_full_text,
    )


if __name__ == "__main__":
    main()
