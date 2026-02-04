"""Streamlit UI for NPDR AI Literature Tracker - Enhanced Version."""

import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from trial_lit_intel.term_expander import expand_clinical_term
from trial_lit_intel.snowflake_search import search_clinical_ai_literature, AI_TERMS
from trial_lit_intel.mesh_search import search_pubmed_by_mesh
from trial_lit_intel.relevance_filter import score_article_relevance
from trial_lit_intel.journal_filter import filter_by_journal_quality
from trial_lit_intel.llm_client import get_completion

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


def deduplicate_papers(papers: list) -> list:
    """Remove duplicate papers based on PMID."""
    seen_pmids = set()
    unique_papers = []
    for paper in papers:
        pmid = str(paper.get("pmid", ""))
        if pmid and pmid not in seen_pmids:
            seen_pmids.add(pmid)
            unique_papers.append(paper)
        elif not pmid:
            # Keep papers without PMID (shouldn't happen often)
            unique_papers.append(paper)
    return unique_papers


def generate_research_gaps(papers: list, clinical_term: str) -> str:
    """Use LLM to identify research gaps in the literature."""
    # Summarize what's covered
    methods = [p.get("model_architecture", "") for p in papers if p.get("model_architecture")]
    tasks = [p.get("task_type", "") for p in papers if p.get("task_type")]
    challenges_covered = {
        "long_term": sum(1 for p in papers if p.get("addresses_long_term_prediction_ch1") == "Y"),
        "early_signals": sum(1 for p in papers if p.get("identifies_early_signals_ch2") == "Y"),
        "low_events": sum(1 for p in papers if p.get("handles_low_event_rates_ch3") == "Y"),
        "rapid_prog": sum(1 for p in papers if p.get("identifies_rapid_progressors_ch4") == "Y"),
        "grading": sum(1 for p in papers if p.get("improves_grading_consistency_ch5") == "Y"),
    }

    prompt = f"""Analyze this summary of {len(papers)} AI/ML papers on {clinical_term}:

Methods used: {Counter(methods).most_common(10)}
Task types: {Counter(tasks).most_common(10)}
Challenge coverage:
- Long-term prediction: {challenges_covered['long_term']} papers
- Early signal detection: {challenges_covered['early_signals']} papers
- Low event rate handling: {challenges_covered['low_events']} papers
- Rapid progressor identification: {challenges_covered['rapid_prog']} papers
- Grading consistency: {challenges_covered['grading']} papers

Based on this, identify 3-5 key RESEARCH GAPS - areas that are understudied or missing.
Be specific and actionable. Format as bullet points."""

    try:
        return get_completion(prompt, model="haiku")
    except Exception as e:
        return f"Error generating research gaps: {e}"


def generate_trend_analysis(papers: list) -> str:
    """Analyze trends in the literature over time."""
    # Group by year
    by_year = {}
    for p in papers:
        year = p.get("year", "Unknown")
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(p)

    # Get methods by year
    trends = {}
    for year, year_papers in sorted(by_year.items()):
        methods = [p.get("model_architecture", "Unknown") for p in year_papers]
        trends[year] = Counter(methods).most_common(3)

    prompt = f"""Analyze these trends in AI/ML methods for diabetic retinopathy over time:

{trends}

Identify:
1. Which methods are INCREASING in popularity (trending up)
2. Which methods are DECLINING (older approaches)
3. Any emerging techniques appearing in recent years
4. Overall trajectory of the field

Be concise - 3-4 bullet points total."""

    try:
        return get_completion(prompt, model="haiku")
    except Exception as e:
        return f"Error generating trend analysis: {e}"


def cluster_papers(papers: list) -> dict:
    """Cluster papers by methodology/approach."""
    clusters = {
        "CNN-based": [],
        "Transformer/ViT": [],
        "Traditional ML": [],
        "Ensemble/Hybrid": [],
        "Other/Unknown": [],
    }

    for paper in papers:
        arch = (paper.get("model_architecture", "") or "").lower()

        if any(x in arch for x in ["transformer", "vit", "vision transformer", "bert", "attention"]):
            clusters["Transformer/ViT"].append(paper)
        elif any(x in arch for x in ["cnn", "resnet", "vgg", "inception", "densenet", "efficientnet", "convolution"]):
            clusters["CNN-based"].append(paper)
        elif any(x in arch for x in ["random forest", "svm", "xgboost", "logistic", "decision tree", "gradient boost"]):
            clusters["Traditional ML"].append(paper)
        elif any(x in arch for x in ["ensemble", "hybrid", "multi", "combined"]):
            clusters["Ensemble/Hybrid"].append(paper)
        else:
            clusters["Other/Unknown"].append(paper)

    return clusters


def main():
    st.title("🔬 NPDR AI Literature Tracker")
    st.markdown("*Structured extraction for diabetic retinopathy AI/ML literature*")

    # Get current year for date filters
    current_year = datetime.now().year

    # Sidebar
    with st.sidebar:
        st.header("Pipeline Settings")

        st.markdown("**Search Method**")
        search_method = st.radio(
            "Term expansion",
            options=["mesh", "text"],
            index=0,
            format_func=lambda x: "MeSH ontology (snek)" if x == "mesh" else "Text-based (LLM)",
            help="MeSH uses standardized medical ontology via snek. Text uses LLM-generated synonyms.",
        )

        max_results = st.slider(
            "Max results",
            min_value=50,
            max_value=1000,
            value=500,
            step=50,
        )

        min_relevance = st.slider(
            "Min relevance score",
            min_value=1,
            max_value=10,
            value=7,
        )

        st.markdown("---")
        st.markdown("**Date Range Filter**")

        year_range = st.slider(
            "Publication years",
            min_value=2010,
            max_value=current_year,
            value=(2018, current_year),
            help="Filter papers by publication year",
        )

        st.markdown("---")
        st.markdown("**Journal Quality Filter**")

        journal_tier = st.radio(
            "Journal tier threshold",
            options=[1, 2],
            index=1,
            format_func=lambda x: "Tier 1 only (top journals)" if x == 1 else "Tier 1 + 2 (more journals)",
            help="Tier 1: Nature Medicine, JAMA, Lancet, etc. Tier 2: Scientific Reports, European Radiology, etc.",
        )

        st.markdown("---")
        st.markdown("**Full Text Retrieval** *(Optional - slow)*")

        enable_pmc = st.checkbox(
            "Enable PMC full text",
            value=False,
            help="Fetch full text from PubMed Central (~3-5% yield)",
        )

        enable_biorxiv = st.checkbox(
            "Enable bioRxiv search",
            value=False,
            help="Search for preprints on bioRxiv/medRxiv (slow)",
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

        st.markdown("---")
        st.markdown("**AI Analysis** *(runs after extraction)*")

        enable_gap_analysis = st.checkbox(
            "Research gap analysis",
            value=True,
            help="LLM identifies understudied areas",
        )

        enable_trend_analysis = st.checkbox(
            "Trend analysis",
            value=True,
            help="Analyze method trends over time",
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
        1. Map to MeSH (snek) → 2. Search PubMed → 3. Deduplicate
        4. Filter quality → 5. Score relevance → 6. Extract data
        """)

    # Run pipeline
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        if not clinical_term:
            st.error("Please enter a clinical term.")
            return

        progress_bar = st.progress(0)
        status = st.empty()
        paper_progress = st.empty()

        try:
            # Determine total steps
            total_steps = 6
            if enable_pmc:
                total_steps += 1
            if enable_biorxiv:
                total_steps += 1
            if enable_gap_analysis:
                total_steps += 1
            if enable_trend_analysis:
                total_steps += 1
            current_step = 0

            def next_step(label):
                nonlocal current_step
                current_step += 1
                status.text(f"Step {current_step}/{total_steps}: {label}")
                progress_bar.progress(int(current_step / total_steps * 95))

            # Step 1 & 2: Search based on method
            if search_method == "mesh":
                next_step("Mapping to MeSH terms via snek...")
                next_step("Searching PubMed by MeSH terms...")
                articles = search_pubmed_by_mesh(clinical_term, max_results=max_results, min_year=year_range[0])
                st.info(f"Found {len(articles)} articles via MeSH search")
            else:
                next_step("Expanding clinical term...")
                terms = expand_clinical_term(clinical_term)
                st.info(f"Expanded to {len(terms)} terms: {', '.join(terms)}")

                next_step("Searching PubMed via Snowflake...")
                articles = search_clinical_ai_literature(terms, max_results_per_query=max_results // 15)
                st.info(f"Found {len(articles)} articles")

            if not articles:
                st.warning("No articles found. Try adjusting your search term.")
                return

            # Step 3: Deduplicate
            next_step("Removing duplicates...")
            original_count = len(articles)
            articles = deduplicate_papers(articles)
            if original_count != len(articles):
                st.info(f"Removed {original_count - len(articles)} duplicates → {len(articles)} unique articles")

            # Step 4: Filter by year range (if not already done by MeSH search)
            if search_method != "mesh":
                articles = [a for a in articles if year_range[0] <= int(a.get("year", 0) or 0) <= year_range[1]]
                st.info(f"{len(articles)} articles in year range {year_range[0]}-{year_range[1]}")

            # Step 5: Filter by journal quality
            next_step("Filtering by journal quality...")
            articles = filter_by_journal_quality(articles, tier_threshold=journal_tier)
            st.info(f"{len(articles)} articles from quality journals")

            # Step 6: Filter by relevance
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

            # Optional: Fetch full text
            full_texts = {}
            pmids = [str(p.get("pmid", "")) for p in articles]

            if enable_pmc:
                next_step("Fetching PMC full text...")
                pmc_results = batch_fetch_pmc_full_text(pmids, max_workers=max_workers)
                for pmid, result in pmc_results.items():
                    if result:
                        full_texts[pmid] = result
                st.info(f"Found {len(full_texts)}/{len(articles)} papers in PMC")

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
                                "source": "bioRxiv",
                                "is_preprint": True,
                                "note": "Data from preprint",
                            }
                            biorxiv_count += 1
                    st.info(f"Found {biorxiv_count} preprints on bioRxiv/medRxiv")

            # Extract structured data with progress
            next_step("Extracting structured data...")

            # Custom extraction with progress display
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from npdr_pipeline.extractor import NPDRExtractor

            extractor = NPDRExtractor()
            extract_workers = max_workers * 2 if not full_texts else max_workers

            def extract_single(paper):
                pmid = str(paper.get("pmid", ""))
                ft = full_texts.get(pmid, {})
                if ft and ft.get("full_text"):
                    return extractor.extract_from_full_text(paper, ft["full_text"])
                return extractor.extract_from_abstract(paper)

            extracted = []
            completed = 0

            with ThreadPoolExecutor(max_workers=extract_workers) as executor:
                futures = {executor.submit(extract_single, p): p for p in articles}

                for future in as_completed(futures):
                    completed += 1
                    paper = futures[future]
                    try:
                        result = future.result()
                        paper_with_data = paper.copy()
                        paper_with_data.update(result)
                        extracted.append(paper_with_data)
                    except Exception as e:
                        paper_copy = paper.copy()
                        paper_copy["extraction_error"] = str(e)
                        extracted.append(paper_copy)

                    # Update progress
                    paper_progress.text(f"Extracting: {completed}/{len(articles)} papers")
                    progress_bar.progress(int((current_step - 1 + completed / len(articles)) / total_steps * 95))

            paper_progress.empty()

            # Mark papers needing manual review
            for paper in extracted:
                source = paper.get("data_source", "")
                if "Abstract" in source:
                    critical_missing = any(
                        paper.get(field, "Not reported") in ["Not reported", "N/A", "Extraction failed"]
                        for field in ["dataset_size", "model_architecture", "external_validation"]
                    )
                    paper["needs_manual_review"] = "Yes" if critical_missing else "No"
                else:
                    paper["needs_manual_review"] = "No"

            # AI Analysis
            gap_analysis = None
            trend_analysis = None

            if enable_gap_analysis:
                next_step("Analyzing research gaps...")
                gap_analysis = generate_research_gaps(extracted, clinical_term)

            if enable_trend_analysis:
                next_step("Analyzing trends...")
                trend_analysis = generate_trend_analysis(extracted)

            progress_bar.progress(100)
            status.text("Pipeline complete!")

            # Store results
            st.session_state["npdr_results"] = extracted
            st.session_state["clinical_term"] = clinical_term
            st.session_state["gap_analysis"] = gap_analysis
            st.session_state["trend_analysis"] = trend_analysis
            st.session_state["year_range"] = year_range  # Store initial year range

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

        # ============ FILTERS & SORTING ============
        st.subheader("Filter & Sort")
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            # Year filter for results - use the initial year range from sidebar
            initial_year_range = st.session_state.get("year_range", (2010, current_year))
            year_options = list(range(initial_year_range[0], initial_year_range[1] + 1))
            if year_options:
                year_filter = st.select_slider(
                    "Filter by year",
                    options=year_options,
                    value=(initial_year_range[0], initial_year_range[1]),
                )
            else:
                year_filter = (0, 9999)

        with filter_col2:
            sort_option = st.selectbox(
                "Sort by",
                options=["Year (newest)", "Year (oldest)", "Relevance (high)", "Journal Tier"],
            )

        with filter_col3:
            relevance_filter = st.multiselect(
                "Relevance to BLKR-201",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
            )

        # Apply filters
        filtered = [p for p in extracted
                    if year_filter[0] <= int(p.get("year", 0) or 0) <= year_filter[1]
                    and p.get("relevance_to_blkr201", "Unknown") in relevance_filter + ["Unknown"]]

        # Apply sorting
        if sort_option == "Year (newest)":
            filtered.sort(key=lambda x: int(x.get("year", 0) or 0), reverse=True)
        elif sort_option == "Year (oldest)":
            filtered.sort(key=lambda x: int(x.get("year", 0) or 0))
        elif sort_option == "Relevance (high)":
            relevance_order = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
            filtered.sort(key=lambda x: relevance_order.get(x.get("relevance_to_blkr201", "Unknown"), 3))
        elif sort_option == "Journal Tier":
            filtered.sort(key=lambda x: x.get("journal_tier", 99))

        st.caption(f"Showing {len(filtered)} of {len(extracted)} papers after filters")

        # ============ VISUALIZATIONS ============
        st.subheader("📊 Visualizations")

        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Papers by Year", "Journal Distribution", "Challenge Heatmap", "Method Clusters"
        ])

        with viz_tab1:
            # Papers by year bar chart
            year_counts = Counter(int(p.get("year", 0) or 0) for p in filtered if p.get("year"))
            year_df = pd.DataFrame([
                {"Year": year, "Count": count}
                for year, count in sorted(year_counts.items()) if year > 0
            ])
            if not year_df.empty:
                fig = px.bar(year_df, x="Year", y="Count", title="Publications by Year")
                fig.update_layout(xaxis_tickmode='linear')
                st.plotly_chart(fig, use_container_width=True)

        with viz_tab2:
            # Journal distribution pie chart
            journal_counts = Counter(p.get("journal", "Unknown") for p in filtered)
            top_journals = dict(journal_counts.most_common(10))
            if top_journals:
                fig = px.pie(
                    values=list(top_journals.values()),
                    names=list(top_journals.keys()),
                    title="Top 10 Journals"
                )
                st.plotly_chart(fig, use_container_width=True)

        with viz_tab3:
            # Challenge coverage heatmap
            challenges = [
                ("Long-term Prediction", "addresses_long_term_prediction_ch1"),
                ("Early Signals", "identifies_early_signals_ch2"),
                ("Low Event Rates", "handles_low_event_rates_ch3"),
                ("Rapid Progressors", "identifies_rapid_progressors_ch4"),
                ("Grading Consistency", "improves_grading_consistency_ch5"),
            ]

            heatmap_data = []
            for name, field in challenges:
                yes = sum(1 for p in filtered if p.get(field) == "Y")
                partial = sum(1 for p in filtered if p.get(field) == "Partial")
                no = sum(1 for p in filtered if p.get(field) == "N")
                heatmap_data.append([yes, partial, no])

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=["Yes", "Partial", "No"],
                y=[c[0] for c in challenges],
                colorscale="Blues",
                text=heatmap_data,
                texttemplate="%{text}",
                textfont={"size": 14},
            ))
            fig.update_layout(title="Challenge Coverage Heatmap", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with viz_tab4:
            # Method clusters
            clusters = cluster_papers(filtered)
            cluster_df = pd.DataFrame([
                {"Method Category": name, "Count": len(papers)}
                for name, papers in clusters.items() if papers
            ])
            if not cluster_df.empty:
                fig = px.bar(cluster_df, x="Method Category", y="Count", title="Papers by Method Type")
                st.plotly_chart(fig, use_container_width=True)

        # ============ AI INSIGHTS ============
        st.subheader("🤖 AI Insights")

        insight_tab1, insight_tab2 = st.tabs(["Research Gaps", "Trend Analysis"])

        with insight_tab1:
            if st.session_state.get("gap_analysis"):
                st.markdown(st.session_state["gap_analysis"])
            else:
                st.info("Enable 'Research gap analysis' in sidebar and re-run pipeline")

        with insight_tab2:
            if st.session_state.get("trend_analysis"):
                st.markdown(st.session_state["trend_analysis"])
            else:
                st.info("Enable 'Trend analysis' in sidebar and re-run pipeline")

        # ============ SUMMARY STATS ============
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pmc_count = sum(1 for p in filtered if "PMC" in p.get("data_source", ""))
            st.metric("PMC Full Text", pmc_count)

        with col2:
            preprint_count = sum(1 for p in filtered if "Preprint" in p.get("data_source", ""))
            st.metric("Preprints", preprint_count)

        with col3:
            abstract_only = sum(1 for p in filtered if p.get("data_source", "") == "Abstract")
            st.metric("Abstract Only", abstract_only)

        with col4:
            needs_review = sum(1 for p in filtered if p.get("needs_manual_review") == "Yes")
            st.metric("Needs Review", needs_review)

        # ============ INTERACTIVE CHALLENGE TABLE ============
        st.subheader("Challenge Coverage")
        st.caption("Click on any number to view matching papers")

        challenges = [
            ("Ch1: Long-term Prediction", "addresses_long_term_prediction_ch1"),
            ("Ch2: Early Signals", "identifies_early_signals_ch2"),
            ("Ch3: Low Event Rates", "handles_low_event_rates_ch3"),
            ("Ch4: Rapid Progressors", "identifies_rapid_progressors_ch4"),
            ("Ch5: Grading Consistency", "improves_grading_consistency_ch5"),
        ]

        if "challenge_filter" not in st.session_state:
            st.session_state["challenge_filter"] = None

        header_cols = st.columns([3, 1, 1, 1])
        header_cols[0].markdown("**Challenge**")
        header_cols[1].markdown("**Yes**")
        header_cols[2].markdown("**Partial**")
        header_cols[3].markdown("**No**")

        for name, field in challenges:
            yes_papers = [p for p in filtered if p.get(field) == "Y"]
            partial_papers = [p for p in filtered if p.get(field) == "Partial"]
            no_papers = [p for p in filtered if p.get(field) == "N"]

            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"{name}")

            if cols[1].button(f"{len(yes_papers)}", key=f"{field}_yes", disabled=len(yes_papers) == 0):
                st.session_state["challenge_filter"] = {
                    "name": name, "status_label": "Yes", "papers": yes_papers
                }

            if cols[2].button(f"{len(partial_papers)}", key=f"{field}_partial", disabled=len(partial_papers) == 0):
                st.session_state["challenge_filter"] = {
                    "name": name, "status_label": "Partial", "papers": partial_papers
                }

            if cols[3].button(f"{len(no_papers)}", key=f"{field}_no", disabled=len(no_papers) == 0):
                st.session_state["challenge_filter"] = {
                    "name": name, "status_label": "No", "papers": no_papers
                }

        # Display filtered papers
        if st.session_state.get("challenge_filter"):
            filter_info = st.session_state["challenge_filter"]
            papers_to_show = filter_info["papers"]

            st.markdown("---")
            st.subheader(f"📋 {filter_info['name']} - {filter_info['status_label']} ({len(papers_to_show)} papers)")

            if st.button("✕ Clear filter"):
                st.session_state["challenge_filter"] = None
                st.rerun()

            for paper in papers_to_show:
                pmid = paper.get("pmid", "")
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                title = paper.get("title", "No title")
                overview = paper.get("key_findings", "") or paper.get("relevance_reason", "") or "No overview available"

                st.markdown(f"""
**{title}**

{overview}

🔗 [View on PubMed]({pubmed_url})

---
""")

        # ============ EXPORT ============
        st.subheader("Export")
        csv_data = create_csv_download(filtered)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            label="📥 Download CSV (Tracker Format)",
            data=csv_data,
            file_name=f"npdr_tracker_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ============ PAPERS LIST ============
        st.subheader("Extracted Papers")

        for i, paper in enumerate(filtered[:50], 1):
            relevance = paper.get("relevance_to_blkr201", "Unknown")
            relevance_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(relevance, "⚪")
            source = paper.get("data_source", "Unknown")
            needs_review_icon = "⚠️" if paper.get("needs_manual_review") == "Yes" else ""
            year = paper.get("year", "N/A")

            with st.expander(f"{i}. [{year}] {paper.get('title', 'No title')[:70]}... {relevance_emoji} {needs_review_icon}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**First Author:** {paper.get('first_author', 'N/A')}")
                    st.markdown(f"**Year:** {year}")
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

                pmid = paper.get("pmid", "")
                if pmid:
                    st.markdown(f"🔗 [View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

        if len(filtered) > 50:
            st.info(f"Showing first 50 of {len(filtered)} papers. Download CSV for complete list.")


if __name__ == "__main__":
    main()
