"""CLI entry point for Trial-Lit-Intel."""

import argparse
import json
import sys

from .term_expander import expand_clinical_term


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered clinical literature search tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m trial_lit_intel.main "diabetic retinopathy"
  python -m trial_lit_intel.main "diabetic retinopathy" --filter-relevance
  python -m trial_lit_intel.main "diabetic retinopathy" --filter-relevance --analyze-problems
  python -m trial_lit_intel.main "diabetic retinopathy" --analyze-problems --output results.json
  python -m trial_lit_intel.main "lung cancer" --max-results 100 --analyze-problems
  python -m trial_lit_intel.main "parkinson's" --use-ncbi
        """,
    )
    parser.add_argument(
        "term",
        help="Clinical term to search (e.g., 'diabetic retinopathy')",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout as JSON)",
    )
    parser.add_argument(
        "--max-results", "-m",
        type=int,
        default=50,
        help="Maximum results per search query (default: 50)",
    )
    parser.add_argument(
        "--skip-expansion",
        action="store_true",
        help="Skip synonym expansion, search only the exact term",
    )
    parser.add_argument(
        "--use-ncbi",
        action="store_true",
        help="Use NCBI API instead of Snowflake (slower, but no Snowflake access needed)",
    )
    parser.add_argument(
        "--filter-relevance",
        action="store_true",
        help="Use LLM to filter articles by relevance (reads abstracts, keeps only highly relevant)",
    )
    parser.add_argument(
        "--min-relevance",
        type=int,
        default=7,
        help="Minimum relevance score (0-10) when using --filter-relevance (default: 7)",
    )
    parser.add_argument(
        "--analyze-problems",
        action="store_true",
        help="Classify papers by research problem and show heatmap of literature density",
    )

    args = parser.parse_args()

    # Import the appropriate search module
    if args.use_ncbi:
        from .pubmed_search import search_clinical_ai_literature, AI_TERMS
        data_source = "NCBI API"
    else:
        from .snowflake_search import search_clinical_ai_literature, AI_TERMS
        data_source = "Snowflake"

    print(f"=== Trial-Lit-Intel ===")
    print(f"Clinical term: {args.term}")
    print(f"Data source: {data_source}\n")

    # Step 1: Expand clinical term into synonyms
    if args.skip_expansion:
        clinical_terms = [args.term]
        print(f"Skipping expansion, using only: {args.term}\n")
    else:
        print("Step 1: Expanding clinical term into synonyms...")
        clinical_terms = expand_clinical_term(args.term)
        print(f"Found {len(clinical_terms)} terms:")
        for t in clinical_terms:
            print(f"  - {t}")
        print()

    # Step 2: Search PubMed
    print(f"Step 2: Searching PubMed for AI/ML literature via {data_source}...")
    print(f"AI terms being searched: {', '.join(AI_TERMS)}\n")

    articles = search_clinical_ai_literature(
        clinical_terms,
        max_results_per_query=args.max_results,
    )

    # Step 3: Filter by relevance (optional)
    if args.filter_relevance:
        print(f"\nStep 3: Filtering articles by relevance using LLM...")
        from .relevance_filter import score_article_relevance

        articles = score_article_relevance(
            articles,
            clinical_term=args.term,
            min_score=args.min_relevance,
        )

    # Step 4: Analyze research problems (optional)
    category_summary = None
    research_gaps = None
    if args.analyze_problems:
        step_num = 4 if args.filter_relevance else 3
        print(f"\nStep {step_num}: Classifying papers by research problem...")
        from .problem_classifier import (
            classify_papers_by_problem,
            generate_heatmap_data,
            print_heatmap_ascii,
            identify_research_gaps,
        )

        articles, category_summary = classify_papers_by_problem(
            articles,
            clinical_term=args.term,
        )

        # Generate and display heatmap
        heatmap_data = generate_heatmap_data(category_summary)
        print_heatmap_ascii(heatmap_data)

        # Identify research gaps
        research_gaps = identify_research_gaps(category_summary)
        if research_gaps:
            print("\n📊 POTENTIAL RESEARCH GAPS (Sparse Areas):")
            print("-" * 50)
            for gap in research_gaps:
                severity = "🔴" if gap['gap_severity'] == 'high' else "🟡"
                print(f"{severity} {gap['category_name']}: {gap['count']} papers ({gap['percentage']}%)")
                print(f"   {gap['description']}")

    # Step 5: Output results
    print(f"\n=== Results ===")
    print(f"Total unique articles found: {len(articles)}\n")

    result = {
        "query": {
            "original_term": args.term,
            "expanded_terms": clinical_terms,
            "ai_terms": AI_TERMS,
        },
        "total_articles": len(articles),
        "articles": articles,
    }

    # Add problem analysis to output if performed
    if category_summary:
        result["problem_analysis"] = {
            "category_summary": category_summary,
            "research_gaps": research_gaps,
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        # Print summary to stdout
        print("Top 10 articles:")
        for i, article in enumerate(articles[:10], 1):
            # Show relevance score if available
            score_str = ""
            if 'relevance_score' in article:
                score_str = f" [Relevance: {article['relevance_score']}/10]"

            print(f"\n{i}. {article['title']}{score_str}")
            print(f"   Authors: {', '.join(article['authors'][:3])}{'...' if len(article['authors']) > 3 else ''}")
            print(f"   Journal: {article['journal']} ({article['year']})")
            print(f"   URL: {article['url']}")

            # Show relevance reason if available
            if article.get('relevance_reason'):
                print(f"   Why relevant: {article['relevance_reason']}")

        if len(articles) > 10:
            print(f"\n... and {len(articles) - 10} more articles.")
            print("Use --output to save full results to a file.")


if __name__ == "__main__":
    main()
