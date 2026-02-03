"""Streamlit UI for Trial-Lit-Intel."""

import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import io
from trial_lit_intel.term_expander import expand_clinical_term
from trial_lit_intel.snowflake_search import search_clinical_ai_literature, AI_TERMS
from trial_lit_intel.relevance_filter import score_article_relevance
from trial_lit_intel.problem_classifier import (
    classify_papers_by_problem,
    generate_heatmap_data,
    identify_research_gaps,
    DEFAULT_PROBLEM_FRAMEWORKS,
)
from trial_lit_intel.metadata_extractor import extract_paper_metadata
from trial_lit_intel.journal_filter import filter_by_journal_quality

# Page config
st.set_page_config(
    page_title="Trial-Lit-Intel",
    page_icon="📚",
    layout="wide",
)

# Example clinical terms
EXAMPLE_TERMS = [
    "diabetic retinopathy",
    "lung cancer",
    "breast cancer",
    "Alzheimer's disease",
    "Parkinson's disease",
    "age-related macular degeneration",
    "glaucoma",
    "heart failure",
    "sepsis",
    "COVID-19",
]


def create_heatmap_chart(heatmap_data: dict):
    """Create a horizontal bar chart for the heatmap."""
    if not heatmap_data['categories']:
        return None

    df = pd.DataFrame({
        'Category': heatmap_data['categories'],
        'Papers': heatmap_data['counts'],
        'Percentage': heatmap_data['percentages'],
        'Density': heatmap_data['density_levels'],
    })

    # Color mapping
    color_map = {
        'dense': '#2E7D32',     # Green
        'moderate': '#F9A825',  # Yellow/Orange
        'sparse': '#C62828',    # Red
    }

    df['Color'] = df['Density'].map(color_map)

    return df


def create_csv_download(articles: list[dict]) -> bytes:
    """Create CSV file for download."""
    rows = []
    for article in articles:
        rows.append({
            'Title': article.get('title', ''),
            'Year': article.get('year', ''),
            'Journal': article.get('journal', ''),
            'Journal Tier': article.get('journal_tier', ''),
            'Authors': ', '.join(article.get('authors', [])[:5]),
            'PMID': article.get('pmid', ''),
            'URL': article.get('url', ''),
            'Relevance Score': article.get('relevance_score', ''),
            'Primary Problem': article.get('primary_problem', ''),
            'Problem Categories': ', '.join(article.get('problem_categories', [])),
            'External Validation': article.get('external_validation', ''),
            'Public Code': article.get('public_code', ''),
            'Sample Size': article.get('sample_size', ''),
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')


def main():
    st.title("📚 Trial-Lit-Intel")
    st.markdown("*AI-powered clinical literature search and gap analysis*")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Settings")

        # Max results
        max_results = st.slider(
            "Max results per query",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )

        # Minimum relevance score
        min_relevance = st.slider(
            "Minimum relevance score",
            min_value=1,
            max_value=10,
            value=7,
        )

        st.markdown("---")
        st.markdown("**Quality Filters**")

        # Journal quality filter
        use_journal_filter = st.checkbox(
            "Filter by journal quality",
            value=True,
            help="Only include articles from top medical AI journals",
        )

        if use_journal_filter:
            journal_tier = st.radio(
                "Journal tier:",
                options=[1, 2],
                index=1,
                format_func=lambda x: "Tier 1 only (highest impact)" if x == 1 else "Tier 1 + 2 (top 20 journals)",
                help="Tier 1: Nature Medicine, Lancet Digital Health, NEJM AI, etc.\nTier 2: Includes specialty journals like Eur Radiology, Br J Ophthalmol",
            )
        else:
            journal_tier = 2

        st.markdown("---")
        st.markdown("**Problem Categories**")

        # Category selection
        selected_categories = {}
        for cat_id, cat_info in DEFAULT_PROBLEM_FRAMEWORKS.items():
            selected = st.checkbox(
                cat_info['name'],
                value=True,
                key=f"cat_{cat_id}",
                help=cat_info['description'],
            )
            if selected:
                selected_categories[cat_id] = cat_info

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enter Clinical Term")

        # Term input
        input_method = st.radio(
            "Choose input method:",
            ["Select from examples", "Enter custom term"],
            horizontal=True,
        )

        if input_method == "Select from examples":
            clinical_term = st.selectbox(
                "Select a clinical term:",
                options=EXAMPLE_TERMS,
            )
        else:
            clinical_term = st.text_input(
                "Enter your clinical term:",
                placeholder="e.g., diabetic retinopathy",
            )

    with col2:
        st.subheader("Quick Info")
        st.markdown(f"""
        - **AI Terms:** {', '.join(AI_TERMS[:3])}...
        - **Categories:** {len(selected_categories)} selected
        - **Max Results:** {max_results}
        """)

    # Search button
    if st.button("🔍 Search & Analyze", type="primary", use_container_width=True):
        if not clinical_term:
            st.error("Please enter a clinical term.")
            return

        if not selected_categories:
            st.error("Please select at least one problem category.")
            return

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Calculate total steps based on options
        total_steps = 5 if not use_journal_filter else 6

        try:
            # Step 1: Expand terms
            status_text.text(f"Step 1/{total_steps}: Expanding clinical term...")
            progress_bar.progress(10)

            expanded_terms = expand_clinical_term(clinical_term)
            st.info(f"Expanded to {len(expanded_terms)} terms: {', '.join(expanded_terms)}")

            # Step 2: Search PubMed
            status_text.text(f"Step 2/{total_steps}: Searching PubMed via Snowflake...")
            progress_bar.progress(20)

            articles = search_clinical_ai_literature(
                expanded_terms,
                max_results_per_query=max_results,
            )
            st.info(f"Found {len(articles)} articles")

            if not articles:
                st.warning("No articles found. Try a different search term.")
                progress_bar.progress(100)
                status_text.text("Complete (no results)")
                return

            # Step 3: Filter by journal quality (optional)
            step_num = 3
            if use_journal_filter:
                status_text.text(f"Step {step_num}/{total_steps}: Filtering by journal quality...")
                progress_bar.progress(35)

                articles = filter_by_journal_quality(articles, tier_threshold=journal_tier)
                tier_label = "Tier 1" if journal_tier == 1 else "Tier 1+2"
                st.info(f"{len(articles)} articles from high-quality journals ({tier_label})")

                if not articles:
                    st.warning("No articles from whitelisted journals. Try disabling journal filter.")
                    progress_bar.progress(100)
                    status_text.text("Complete (no quality matches)")
                    return
                step_num += 1

            # Step 4: Filter by relevance
            status_text.text(f"Step {step_num}/{total_steps}: Filtering by relevance...")
            progress_bar.progress(50)

            articles = score_article_relevance(
                articles,
                clinical_term=clinical_term,
                min_score=min_relevance,
            )
            st.info(f"{len(articles)} articles passed relevance filter (score >= {min_relevance})")

            if not articles:
                st.warning("No articles passed the relevance filter. Try lowering the threshold.")
                progress_bar.progress(100)
                status_text.text("Complete (no relevant articles)")
                return
            step_num += 1

            # Step 5: Classify by problem
            status_text.text(f"Step {step_num}/{total_steps}: Classifying by research problem...")
            progress_bar.progress(70)

            articles, category_summary = classify_papers_by_problem(
                articles,
                clinical_term=clinical_term,
                problem_frameworks=selected_categories,
            )
            step_num += 1

            # Step 6: Extract metadata
            status_text.text(f"Step {step_num}/{total_steps}: Extracting paper metadata...")
            progress_bar.progress(85)

            articles = extract_paper_metadata(articles)

            progress_bar.progress(100)
            status_text.text("Analysis complete!")

            # Store results in session state
            st.session_state['articles'] = articles
            st.session_state['category_summary'] = category_summary
            st.session_state['clinical_term'] = clinical_term

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return

    # Display results if available
    if 'articles' in st.session_state and st.session_state['articles']:
        articles = st.session_state['articles']
        category_summary = st.session_state['category_summary']
        clinical_term = st.session_state['clinical_term']

        st.markdown("---")
        st.header(f"Results for: {clinical_term}")

        # Heatmap section
        st.subheader("📊 Research Gap Heatmap")
        st.caption("Click on a category to filter articles")

        # Initialize selected category filter
        if 'selected_category' not in st.session_state:
            st.session_state['selected_category'] = None

        if category_summary:
            heatmap_data = generate_heatmap_data(category_summary)
            df_heatmap = create_heatmap_chart(heatmap_data)

            if df_heatmap is not None:
                # Build a mapping from category name to category id
                cat_name_to_id = {}
                for cat_id, info in category_summary.items():
                    cat_name_to_id[info['name']] = cat_id

                # Display as clickable horizontal bar chart
                for idx, row in df_heatmap.iterrows():
                    cat_name = row['Category']
                    cat_id = cat_name_to_id.get(cat_name, '')
                    is_selected = st.session_state.get('selected_category') == cat_id

                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        # Make category name a clickable button
                        button_label = f"{'→ ' if is_selected else ''}{cat_name}"
                        if st.button(
                            button_label,
                            key=f"cat_btn_{idx}",
                            type="primary" if is_selected else "secondary",
                            use_container_width=True,
                        ):
                            if is_selected:
                                # Clicking again clears the filter
                                st.session_state['selected_category'] = None
                            else:
                                st.session_state['selected_category'] = cat_id
                            st.rerun()
                    with col2:
                        # Create a progress-like bar
                        max_count = df_heatmap['Papers'].max()
                        pct = row['Papers'] / max_count if max_count > 0 else 0
                        st.progress(pct)
                    with col3:
                        density_emoji = {'dense': '🟢', 'moderate': '🟡', 'sparse': '🔴'}
                        st.markdown(f"{row['Papers']} ({row['Percentage']:.1f}%) {density_emoji.get(row['Density'], '')}")

                st.markdown("**Legend:** 🟢 Dense (>60%) | 🟡 Moderate (30-60%) | 🔴 Sparse (<30%)")

            # Research gaps
            gaps = identify_research_gaps(category_summary)
            if gaps:
                st.subheader("🔍 Potential Research Gaps")
                for gap in gaps:
                    severity_emoji = "🔴" if gap['gap_severity'] == 'high' else "🟡"
                    st.markdown(f"{severity_emoji} **{gap['category_name']}**: {gap['count']} papers ({gap['percentage']}%)")
                    st.caption(gap['description'])

        # Filter articles by selected category
        display_articles = articles
        selected_cat = st.session_state.get('selected_category')

        if selected_cat:
            display_articles = [
                a for a in articles
                if selected_cat in a.get('problem_categories', [])
            ]
            # Get category name for display
            cat_display_name = category_summary.get(selected_cat, {}).get('name', selected_cat)
            st.info(f"Showing {len(display_articles)} articles in **{cat_display_name}** category. Click the category again to show all.")

        # Articles table
        st.subheader(f"📄 Articles ({len(display_articles)} {'filtered' if selected_cat else 'found'})")

        # Create download button
        csv_data = create_csv_download(display_articles)
        download_filename = f"trial_lit_intel_{clinical_term.replace(' ', '_')}"
        if selected_cat:
            download_filename += f"_{selected_cat}"
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"{download_filename}.csv",
            mime="text/csv",
        )

        # Display articles in expandable sections
        for i, article in enumerate(display_articles[:20], 1):
            with st.expander(f"{i}. {article.get('title', 'No title')[:100]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Year:** {article.get('year', 'N/A')}")
                    journal_tier = article.get('journal_tier', '')
                    tier_badge = f" (Tier {journal_tier})" if journal_tier else ""
                    st.markdown(f"**Journal:** {article.get('journal', 'N/A')}{tier_badge}")
                    st.markdown(f"**PMID:** [{article.get('pmid', 'N/A')}]({article.get('url', '')})")
                with col2:
                    st.markdown(f"**Relevance Score:** {article.get('relevance_score', 'N/A')}/10")
                    st.markdown(f"**External Validation:** {article.get('external_validation', 'unclear')}")
                    st.markdown(f"**Public Code:** {article.get('public_code', 'unclear')}")
                    st.markdown(f"**Sample Size:** {article.get('sample_size', 'not reported')}")

                if article.get('problem_categories'):
                    st.markdown(f"**Categories:** {', '.join(article.get('problem_categories', []))}")

                if article.get('relevance_reason'):
                    st.caption(f"*{article.get('relevance_reason')}*")

        if len(display_articles) > 20:
            st.info(f"Showing first 20 of {len(display_articles)} articles. Download CSV for complete list.")


if __name__ == "__main__":
    main()
