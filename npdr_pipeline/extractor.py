"""LLM-based data extraction for NPDR literature tracker.

Uses Claude Sonnet for structured extraction from abstracts and full text.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Import from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from trial_lit_intel.llm_client import get_completion
except ImportError:
    from .llm_client_local import get_completion


# Model selection
EXTRACTION_MODEL = os.getenv("NPDR_EXTRACTION_MODEL", "sonnet")
ASSESSMENT_MODEL = os.getenv("NPDR_ASSESSMENT_MODEL", "sonnet")

# Extraction schema for NPDR tracker
EXTRACTION_FIELDS = {
    "methodology": [
        "task_type",
        "imaging_modality",
        "other_input_features",
        "prediction_target",
        "prediction_horizon",
        "model_architecture",
        "training_dataset",
        "dataset_size",
        "external_validation",
        "primary_metric",
        "primary_metric_value",
        "secondary_metrics",
    ],
    "challenges": [
        "addresses_long_term_prediction_ch1",
        "identifies_early_signals_ch2",
        "handles_low_event_rates_ch3",
        "identifies_rapid_progressors_ch4",
        "improves_grading_consistency_ch5",
    ],
    "synthesis": [
        "relevance_to_blkr201",
        "potential_application",
        "key_findings",
        "limitations",
    ],
}


METHODOLOGY_EXTRACTION_PROMPT = """You are an expert at extracting structured information from scientific papers about AI/ML applications in diabetic retinopathy.

Extract the following fields from the paper text. If information is not available, use "Not reported" or "N/A".

FIELDS TO EXTRACT:
1. task_type: The main ML task (e.g., "Time-to-DR progression", "DRSS Step Change Prediction", "Risk prediction", "DRSS Grading", "Segmentation")
2. imaging_modality: Type of imaging used (e.g., "Color Fundus (2-field)", "UWF CFP", "OCTA", "OCT")
3. other_input_features: Non-imaging features used (e.g., "Age, gender, HbA1c, BP" or "None (images only)")
4. prediction_target: What the model predicts (e.g., "≥2-step ETDRS DRSS worsening", "Time to any DR progression")
5. prediction_horizon: Time frame of prediction (e.g., "1-5 years", "6, 12, 24 months", "N/A for cross-sectional")
6. model_architecture: ML/DL architecture used (e.g., "ResNet-50 + self-attention", "Inception-v3", "Cox regression", "Random Forest")
7. training_dataset: Dataset(s) used for training (e.g., "RIDE/RISE Phase 3 trials", "UK Biobank")
8. dataset_size: Sample size with details (e.g., "717K pretrain; 19,100 dev; 10,768 external" or "530 patients, 680 eyes")
9. external_validation: Whether externally validated and details (e.g., "Yes (8 cohorts, multiethnic)" or "No (5-fold CV only)")
10. primary_metric: Main performance metric (e.g., "C-index", "AUC", "AUPRC", "HR")
11. primary_metric_value: Value of primary metric (e.g., "0.823", "0.79 ± 0.05")
12. secondary_metrics: Other reported metrics (e.g., "Sens 91%, Spec 65%; IBS 0.153")

Return ONLY a JSON object with these exact field names. Example:
{
    "task_type": "DRSS Step Change Prediction",
    "imaging_modality": "Color Fundus (7-field ETDRS)",
    "other_input_features": "None (images only)",
    "prediction_target": "≥2-step ETDRS DRSS worsening",
    "prediction_horizon": "6, 12, 24 months",
    "model_architecture": "Inception-v3 DCNNs + Random Forest",
    "training_dataset": "RIDE/RISE Phase 3 trials",
    "dataset_size": "~530 patients; ~680 eyes; ~4,800 CFPs",
    "external_validation": "No (5-fold CV only)",
    "primary_metric": "AUC",
    "primary_metric_value": "0.79 ± 0.05",
    "secondary_metrics": "Sens 91%, Spec 65%"
}"""


CHALLENGE_ASSESSMENT_PROMPT = """You are an expert at assessing the relevance of diabetic retinopathy AI/ML research to specific clinical challenges.

For each challenge below, assess whether the paper addresses it. Answer with:
- "Y" = Yes, directly addresses this challenge
- "N" = No, does not address this challenge
- "Partial" = Partially addresses or tangentially related

CHALLENGES:
1. Ch1 - Long-term Prediction: Does the paper predict outcomes over extended timeframes (1+ years)? Does it model disease trajectory over time?
2. Ch2 - Early Signals: Does it identify early biomarkers or signals before clinical progression? Can it detect risk in early-stage patients?
3. Ch3 - Low Event Rates: Does it address class imbalance or rare progression events? Does it work with limited positive cases?
4. Ch4 - Rapid Progressors: Can it identify patients who will progress quickly? Does it stratify by progression speed?
5. Ch5 - Grading Consistency: Does it address inter-grader variability? Does it improve standardization of DR grading?

Return ONLY a JSON object:
{
    "addresses_long_term_prediction_ch1": "Y/N/Partial",
    "identifies_early_signals_ch2": "Y/N/Partial",
    "handles_low_event_rates_ch3": "Y/N/Partial",
    "identifies_rapid_progressors_ch4": "Y/N/Partial",
    "improves_grading_consistency_ch5": "Y/N/Partial"
}"""


SYNTHESIS_PROMPT = """You are an expert at synthesizing scientific literature for clinical trial planning in diabetic retinopathy.

Context: BLKR-201 is a clinical study investigating treatments for diabetic retinopathy. We need to assess how this paper might inform trial design, patient selection, or endpoint definition.

Provide:
1. relevance_to_blkr201: Rate as "High", "Medium", or "Low" based on:
   - High: Directly applicable to trial design, patient enrichment, or endpoint prediction
   - Medium: Provides useful methodological insights or relevant findings
   - Low: Tangentially related or limited applicability

2. potential_application: How could this paper's methods/findings be applied? (e.g., "Patient enrichment for clinical trials", "DRSS trajectory prediction", "Endpoint selection")

3. key_findings: Summarize the 2-3 most important findings in 1-2 sentences.

4. limitations: List the main limitations of the study.

Return ONLY a JSON object:
{
    "relevance_to_blkr201": "High/Medium/Low",
    "potential_application": "...",
    "key_findings": "...",
    "limitations": "..."
}"""


class NPDRExtractor:
    """Extract structured data from papers for NPDR tracker."""

    def __init__(self, model: str = None):
        self.model = model or EXTRACTION_MODEL

    def extract_from_abstract(self, paper: dict) -> dict:
        """Extract data from abstract only (Tier 1).

        Args:
            paper: Dict with 'title', 'abstract', 'pmid'

        Returns:
            Extracted fields with confidence indicators
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not abstract:
            return self._empty_extraction("No abstract available")

        text = f"Title: {title}\n\nAbstract: {abstract}"

        return self._extract_all_fields(text, source="Abstract")

    def extract_from_full_text(self, paper: dict, full_text: str) -> dict:
        """Extract data from full text (Tier 2/3).

        Args:
            paper: Dict with 'title', 'abstract', 'pmid'
            full_text: Full text content

        Returns:
            Extracted fields with confidence indicators
        """
        title = paper.get("title", "")

        # Truncate full text if too long (keep under ~150K tokens)
        max_chars = 400000  # ~100K tokens
        if len(full_text) > max_chars:
            # Keep beginning and end (methods usually at beginning, results at end)
            half = max_chars // 2
            full_text = full_text[:half] + "\n\n[...TRUNCATED...]\n\n" + full_text[-half:]

        text = f"Title: {title}\n\n{full_text}"

        return self._extract_all_fields(text, source="Full Text")

    def _extract_all_fields(self, text: str, source: str) -> dict:
        """Extract all fields from text.

        Args:
            text: Paper text (abstract or full)
            source: "Abstract" or "Full Text"

        Returns:
            Dict with all extracted fields
        """
        result = {
            "data_source": source,
            "extraction_confidence": "High" if source == "Full Text" else "Medium",
        }

        # Extract methodology fields
        try:
            methodology = self._extract_methodology(text)
            result.update(methodology)
        except Exception as e:
            print(f"  Warning: Methodology extraction failed: {e}")
            result.update(self._empty_methodology())

        # Assess challenges
        try:
            challenges = self._assess_challenges(text)
            result.update(challenges)
        except Exception as e:
            print(f"  Warning: Challenge assessment failed: {e}")
            result.update(self._empty_challenges())

        # Synthesize findings
        try:
            synthesis = self._synthesize(text)
            result.update(synthesis)
        except Exception as e:
            print(f"  Warning: Synthesis failed: {e}")
            result.update(self._empty_synthesis())

        return result

    def _extract_methodology(self, text: str) -> dict:
        """Extract methodology fields."""
        prompt = f"""Paper text:
{text[:50000]}

{METHODOLOGY_EXTRACTION_PROMPT}"""

        response = get_completion(prompt, model=self.model)
        return self._parse_json_response(response, EXTRACTION_FIELDS["methodology"])

    def _assess_challenges(self, text: str) -> dict:
        """Assess challenge relevance."""
        prompt = f"""Paper text:
{text[:30000]}

{CHALLENGE_ASSESSMENT_PROMPT}"""

        response = get_completion(prompt, model=self.model)
        return self._parse_json_response(response, EXTRACTION_FIELDS["challenges"])

    def _synthesize(self, text: str) -> dict:
        """Synthesize findings and assess relevance."""
        prompt = f"""Paper text:
{text[:30000]}

{SYNTHESIS_PROMPT}"""

        response = get_completion(prompt, model=self.model)
        return self._parse_json_response(response, EXTRACTION_FIELDS["synthesis"])

    def _parse_json_response(self, response: str, expected_fields: list) -> dict:
        """Parse JSON response from LLM."""
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        try:
            data = json.loads(response)
            # Ensure all expected fields exist
            for field in expected_fields:
                if field not in data:
                    data[field] = "Not reported"
            return data
        except json.JSONDecodeError:
            # Return empty dict with all fields
            return {field: "Extraction failed" for field in expected_fields}

    def _empty_extraction(self, reason: str) -> dict:
        """Return empty extraction with reason."""
        result = {
            "data_source": "None",
            "extraction_confidence": "None",
            "extraction_note": reason,
        }
        result.update(self._empty_methodology())
        result.update(self._empty_challenges())
        result.update(self._empty_synthesis())
        return result

    def _empty_methodology(self) -> dict:
        return {field: "Not reported" for field in EXTRACTION_FIELDS["methodology"]}

    def _empty_challenges(self) -> dict:
        return {field: "N/A" for field in EXTRACTION_FIELDS["challenges"]}

    def _empty_synthesis(self) -> dict:
        return {field: "Requires manual review" for field in EXTRACTION_FIELDS["synthesis"]}


def batch_extract(
    papers: list,
    full_texts: dict = None,
    max_workers: int = 5,
) -> list:
    """Extract data from multiple papers in parallel.

    Args:
        papers: List of paper dicts with 'pmid', 'title', 'abstract'
        full_texts: Optional dict mapping PMID to full text
        max_workers: Number of parallel workers

    Returns:
        List of papers with extracted fields added
    """
    full_texts = full_texts or {}
    extractor = NPDRExtractor()

    print(f"\nExtracting structured data from {len(papers)} papers...")
    print(f"Model: {EXTRACTION_MODEL}, parallel workers: {max_workers}")

    def extract_paper(paper):
        pmid = str(paper.get("pmid", ""))
        full_text = full_texts.get(pmid, {})

        if full_text and full_text.get("full_text"):
            # Use full text
            extracted = extractor.extract_from_full_text(
                paper, full_text["full_text"]
            )
            if full_text.get("is_preprint"):
                extracted["data_source"] += " (Preprint)"
                extracted["preprint_note"] = full_text.get("note", "")
        else:
            # Use abstract only
            extracted = extractor.extract_from_abstract(paper)

        # Merge with original paper data
        result = paper.copy()
        result.update(extracted)
        return result

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_paper, paper): paper for paper in papers}

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            results.append(result)

            if completed % 5 == 0 or completed == len(papers):
                print(f"  Progress: {completed}/{len(papers)} papers extracted")

    print(f"Extraction complete!")
    return results
