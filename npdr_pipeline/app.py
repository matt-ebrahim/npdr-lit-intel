"""Streamlit UI for NPDR AI Literature Tracker."""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from trial_lit_intel.term_expander import expand_clinical_term
from trial_lit_intel.snowflake_search import search_clinical_ai_literature, AI_TERMS
from trial_lit_intel.relevance_filter import score_article_relevance
from trial_lit_intel.journal_filter import filter_by_journal_quality

from npdr_pipeline.pmc_fetcher import batch_fetch_pmc_full_text
from npdr_pipeline.biorxiv_fetcher import batch_search_biorxiv
from npdr_pipeline.extractor import batch_extract
from npdr_pipeline.npdr_tracker import CSV_COLUMNS, FIELD_TO_COLUMN

# Page config
st.set_page_config(
    page_title="NPDR Literature Tracker",
    page_icon="🔬",
    layout="wide",
)


def create_csv_download(papers: list) -> bytes:
    """Create CSV in tracker format."""
    rows = []
    for paper in papers:
        row = {}
        for field, column in FIELD_TO_COLUMN.items():
            value = paper.get(field, "")
            if field == "url" and not value:
                pmid = paper.get("pmid", "")
                value = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            row[column] = value
        rows.append(row)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    return df.to_csv(index=False).encode("utf-8")


def main():
    st.title("🔬 NPDR AI Literature Tracker")
    st.markdown("*Structured extraction for diabetic retinopathy AI/ML literature*")

    # Sidebar
    with st.sidebar:
        st.header("Pipeline Settings")

        max_results = st.slider(
            "Max results per query",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
        )

        min_relevance = st.slider(
            "Min relevance score",
            min_value=1,
            max_value=10,
            value=7,
        )

        st.markdown("---")
        st.markdown("**Journal Quality Filter**")

        journal_tier = st.radio(
            "Journal tier threshold",
            options=[1, 2],
            index=1,  # Default to tier 2 (both tiers)
            format_func=lambda x: "Tier 1 only (top journals)" if x == 1 else "Tier 1 + 2 (more journals)",
            help="Tier 1: Nature Medicine, JAMA, Lancet, Ophthalmology, etc. Tier 2: Scientific Reports, European Radiology, BJO, etc.",
        )

        st.markdown("---")
        st.markdown("**Full Text Retrieval** *(Optional - slow)*")

        enable_pmc = st.checkbox(
            "Enable PMC full text",
            value=False,
            help="Fetch full text from PubMed Central (~3-5% yield, adds ~5min)",
        )

        enable_biorxiv = st.checkbox(
            "Enable bioRxiv search",
            value=False,
            help="Search for preprints on bioRxiv/medRxiv (slow due to rate limits)",
        )

        st.markdown("---")
        st.markdown("**Extraction Settings**")

        max_workers = st.slider(
            "Parallel workers",
            min_value=1,
            max_value=20,
            value=10,
            help="More workers = faster extraction",
        )

    # Main content
    st.subheader("Search Configuration")

    col1, col2 = st.columns(2)
    with col1:
        clinical_term = st.text_input(
            "Clinical term:",
            value="diabetic retinopathy",
            help="The clinical condition to search for",
        )

    with col2:
        st.markdown("**Pipeline Overview:**")
        st.caption("""
        1. Expand term → 2. Search PubMed → 3. Filter quality
        2. Score relevance → 5. Extract from abstracts
        """)

    # Run pipeline
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        if not clinical_term:
            st.error("Please enter a clinical term.")
            return

        progress_bar = st.progress(0)
        status = st.empty()

        try:
            # Determine total steps based on options
            total_steps = 5
            if enable_pmc:
                total_steps += 1
            if enable_biorxiv:
                total_steps += 1
            current_step = 0

            def next_step(label):
                nonlocal current_step
                current_step += 1
                status.text(f"Step {current_step}/{total_steps}: {label}")
                progress_bar.progress(int(current_step / total_steps * 90))

            # Step 1: Expand terms
            next_step("Expanding clinical term...")
            terms = expand_clinical_term(clinical_term)
            st.info(f"Expanded to {len(terms)} terms: {', '.join(terms)}")

            # Step 2: Search PubMed
            next_step("Searching PubMed via Snowflake...")
            articles = search_clinical_ai_literature(terms, max_results_per_query=max_results)
            st.info(f"Found {len(articles)} articles")

            if not articles:
                st.warning("No articles found. Try adjusting your search term.")
                return

            # Step 3: Filter by journal quality
            next_step("Filtering by journal quality...")
            articles = filter_by_journal_quality(articles, tier_threshold=journal_tier)
            st.info(f"{len(articles)} articles from quality journals")

            # Step 4: Filter by relevance
            next_step("Scoring relevance with LLM...")
            articles = score_article_relevance(
                articles,
                clinical_term=clinical_term,
                min_score=min_relevance,
            )
            st.info(f"{len(articles)} articles passed relevance filter")

            if not articles:
                st.warning("No relevant articles found. Try lowering the relevance threshold.")
                return

            # Prepare for extraction
            for paper in articles:
                authors = paper.get("authors", [])
                paper["first_author"] = authors[0] if authors else "Unknown"

            # Optional: Fetch full text (only if enabled)
            full_texts = {}
            pmids = [str(p.get("pmid", "")) for p in articles]

            if enable_pmc:
                next_step("Fetching PMC full text...")
                pmc_results = batch_fetch_pmc_full_text(pmids, max_workers=max_workers)
                for pmid, result in pmc_results.items():
                    if result:
                        full_texts[pmid] = result

                pmc_count = len(full_texts)
                st.info(f"Found {pmc_count}/{len(articles)} papers in PMC")

            if enable_biorxiv:
                papers_without_ft = [p for p in articles if str(p.get("pmid", "")) not in full_texts]
                if papers_without_ft:
                    next_step("Searching bioRxiv/medRxiv...")
                    biorxiv_results = batch_search_biorxiv(papers_without_ft, max_workers=3)

                    biorxiv_count = 0
                    for pmid, result in biorxiv_results.items():
                        if result and result.get("found"):
                            full_texts[pmid] = {
                                "pmid": pmid,
                                "full_text": result.get("abstract", ""),
                                "source": f"bioRxiv",
                                "is_preprint": True,
                                "note": "Data from preprint; may differ from published version",
                            }
                            biorxiv_count += 1

                    st.info(f"Found {biorxiv_count} preprints on bioRxiv/medRxiv")

            # Final step: Extract structured data
            next_step("Extracting structured data from abstracts...")

            # Use more workers for abstract-only extraction (smaller payloads)
            extract_workers = max_workers * 2 if not full_texts else max_workers
            extracted = batch_extract(articles, full_texts=full_texts, max_workers=extract_workers)

            # Mark papers needing manual review
            for paper in extracted:
                source = paper.get("data_source", "")
                if "Abstract" in source and "Full Text" not in source:
                    critical_missing = any(
                        paper.get(field, "Not reported") in ["Not reported", "N/A", "Extraction failed"]
                        for field in ["dataset_size", "model_architecture", "external_validation"]
                    )
                    paper["needs_manual_review"] = "Yes" if critical_missing else "No"
                else:
                    paper["needs_manual_review"] = "No"

            progress_bar.progress(100)
            status.text("Pipeline complete!")

            # Store results
            st.session_state["npdr_results"] = extracted
            st.session_state["clinical_term"] = clinical_term

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Display results
    if "npdr_results" in st.session_state and st.session_state["npdr_results"]:
        extracted = st.session_state["npdr_results"]
        clinical_term = st.session_state.get("clinical_term", "")

        st.markdown("---")
        st.header(f"Results: {len(extracted)} Papers")

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pmc_count = sum(1 for p in extracted if "PMC" in p.get("data_source", ""))
            st.metric("PMC Full Text", pmc_count)

        with col2:
            preprint_count = sum(1 for p in extracted if "Preprint" in p.get("data_source", ""))
            st.metric("Preprints", preprint_count)

        with col3:
            abstract_only = sum(1 for p in extracted if p.get("data_source", "") == "Abstract")
            st.metric("Abstract Only", abstract_only)

        with col4:
            needs_review = sum(1 for p in extracted if p.get("needs_manual_review") == "Yes")
            st.metric("Needs Review", needs_review)

        # Challenge coverage
        st.subheader("Challenge Coverage")
        challenges = [
            ("Ch1: Long-term Prediction", "addresses_long_term_prediction_ch1"),
            ("Ch2: Early Signals", "identifies_early_signals_ch2"),
            ("Ch3: Low Event Rates", "handles_low_event_rates_ch3"),
            ("Ch4: Rapid Progressors", "identifies_rapid_progressors_ch4"),
            ("Ch5: Grading Consistency", "improves_grading_consistency_ch5"),
        ]

        challenge_data = []
        for name, field in challenges:
            yes = sum(1 for p in extracted if p.get(field) == "Y")
            partial = sum(1 for p in extracted if p.get(field) == "Partial")
            no = sum(1 for p in extracted if p.get(field) == "N")
            challenge_data.append({"Challenge": name, "Yes": yes, "Partial": partial, "No": no})

        df_challenges = pd.DataFrame(challenge_data)
        st.dataframe(df_challenges, use_container_width=True, hide_index=True)

        # Download button
        st.subheader("Export")
        csv_data = create_csv_download(extracted)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            label="📥 Download CSV (Tracker Format)",
            data=csv_data,
            file_name=f"npdr_tracker_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Papers table
        st.subheader("Extracted Papers")

        for i, paper in enumerate(extracted[:20], 1):
            relevance = paper.get("relevance_to_blkr201", "Unknown")
            relevance_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(relevance, "⚪")
            source = paper.get("data_source", "Unknown")
            needs_review = "⚠️" if paper.get("needs_manual_review") == "Yes" else ""

            with st.expander(f"{i}. {paper.get('title', 'No title')[:80]}... {relevance_emoji} {needs_review}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**First Author:** {paper.get('first_author', 'N/A')}")
                    st.markdown(f"**Year:** {paper.get('year', 'N/A')}")
                    st.markdown(f"**Journal:** {paper.get('journal', 'N/A')}")
                    st.markdown(f"**Data Source:** {source}")
                    st.markdown(f"**Relevance:** {relevance}")

                with col2:
                    st.markdown(f"**Task Type:** {paper.get('task_type', 'N/A')}")
                    st.markdown(f"**Imaging:** {paper.get('imaging_modality', 'N/A')}")
                    st.markdown(f"**Model:** {paper.get('model_architecture', 'N/A')}")
                    st.markdown(f"**Dataset Size:** {paper.get('dataset_size', 'N/A')}")
                    st.markdown(f"**External Val:** {paper.get('external_validation', 'N/A')}")

                if paper.get("key_findings"):
                    st.markdown(f"**Key Findings:** {paper.get('key_findings')}")

                if paper.get("potential_application"):
                    st.markdown(f"**Application:** {paper.get('potential_application')}")

                if paper.get("preprint_note"):
                    st.warning(paper.get("preprint_note"))

        if len(extracted) > 20:
            st.info(f"Showing first 20 of {len(extracted)} papers. Download CSV for complete list.")


if __name__ == "__main__":
    main()
