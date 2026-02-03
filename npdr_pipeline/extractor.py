"""LLM-based data extraction for NPDR literature tracker.

OPTIMIZED: Uses single LLM call per paper instead of 3 separate calls.
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
    from trial_lit_intel.llm_client import get_completion, get_completions_parallel_sync
except ImportError:
    from .llm_client_local import get_completion


# Model selection
EXTRACTION_MODEL = os.getenv("NPDR_EXTRACTION_MODEL", "sonnet")

# All extraction fields
ALL_FIELDS = [
    # Methodology (12 fields)
    "task_type", "imaging_modality", "other_input_features", "prediction_target",
    "prediction_horizon", "model_architecture", "training_dataset", "dataset_size",
    "external_validation", "primary_metric", "primary_metric_value", "secondary_metrics",
    # Challenges (5 fields)
    "addresses_long_term_prediction_ch1", "identifies_early_signals_ch2",
    "handles_low_event_rates_ch3", "identifies_rapid_progressors_ch4",
    "improves_grading_consistency_ch5",
    # Synthesis (4 fields)
    "relevance_to_blkr201", "potential_application", "key_findings", "limitations",
]

# Combined extraction prompt (1 call instead of 3)
COMBINED_EXTRACTION_PROMPT = """You are an expert at extracting structured information from scientific papers about AI/ML applications in diabetic retinopathy.

Extract ALL of the following fields from the paper. Return a single JSON object.
If information is not available, use "Not reported" or "N/A".

## METHODOLOGY FIELDS:
1. task_type: Main ML task (e.g., "DRSS Step Change Prediction", "Risk prediction", "Grading")
2. imaging_modality: Imaging type (e.g., "Color Fundus (2-field)", "UWF CFP", "OCTA", "OCT")
3. other_input_features: Non-imaging features (e.g., "Age, gender, HbA1c" or "None (images only)")
4. prediction_target: What model predicts (e.g., "≥2-step DRSS worsening", "DR progression")
5. prediction_horizon: Time frame (e.g., "1-5 years", "12 months", "N/A for cross-sectional")
6. model_architecture: ML/DL architecture (e.g., "ResNet-50", "Inception-v3", "Random Forest")
7. training_dataset: Dataset name (e.g., "RIDE/RISE trials", "UK Biobank")
8. dataset_size: Sample size (e.g., "717K pretrain; 19,100 dev" or "530 patients")
9. external_validation: Validation details (e.g., "Yes (8 cohorts)" or "No (5-fold CV only)")
10. primary_metric: Main metric (e.g., "AUC", "C-index", "AUPRC")
11. primary_metric_value: Metric value (e.g., "0.823", "0.79 ± 0.05")
12. secondary_metrics: Other metrics (e.g., "Sens 91%, Spec 65%")

## CHALLENGE ASSESSMENT (answer Y/N/Partial):
13. addresses_long_term_prediction_ch1: Predicts outcomes 1+ years ahead?
14. identifies_early_signals_ch2: Detects early biomarkers before progression?
15. handles_low_event_rates_ch3: Addresses class imbalance/rare events?
16. identifies_rapid_progressors_ch4: Identifies fast-progressing patients?
17. improves_grading_consistency_ch5: Addresses inter-grader variability?

## SYNTHESIS:
18. relevance_to_blkr201: "High"/"Medium"/"Low" - relevance to clinical trial design
19. potential_application: How to apply findings (e.g., "Patient enrichment", "Endpoint selection")
20. key_findings: 1-2 sentence summary of main findings
21. limitations: Main study limitations

Return ONLY a JSON object with all 21 fields. Example format:
{
    "task_type": "DRSS Step Change Prediction",
    "imaging_modality": "Color Fundus",
    "other_input_features": "None (images only)",
    "prediction_target": "≥2-step DRSS worsening",
    "prediction_horizon": "12 months",
    "model_architecture": "Inception-v3",
    "training_dataset": "RIDE/RISE trials",
    "dataset_size": "530 patients",
    "external_validation": "No (5-fold CV)",
    "primary_metric": "AUC",
    "primary_metric_value": "0.79",
    "secondary_metrics": "Sens 91%, Spec 65%",
    "addresses_long_term_prediction_ch1": "Y",
    "identifies_early_signals_ch2": "Partial",
    "handles_low_event_rates_ch3": "N",
    "identifies_rapid_progressors_ch4": "Y",
    "improves_grading_consistency_ch5": "N",
    "relevance_to_blkr201": "High",
    "potential_application": "Patient enrichment for trials",
    "key_findings": "Model predicts DR progression with 79% AUC using single fundus image.",
    "limitations": "Single-center, no external validation"
}"""


class NPDRExtractor:
    """Extract structured data from papers for NPDR tracker.

    OPTIMIZED: Uses single LLM call per paper (was 3 calls before).
    """

    def __init__(self, model: str = None):
        self.model = model or EXTRACTION_MODEL

    def extract_from_abstract(self, paper: dict) -> dict:
        """Extract data from abstract only (Tier 1)."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        if not abstract:
            return self._empty_extraction("No abstract available")

        text = f"Title: {title}\n\nAbstract: {abstract}"
        return self._extract_all_fields_combined(text, source="Abstract")

    def extract_from_full_text(self, paper: dict, full_text: str) -> dict:
        """Extract data from full text (Tier 2/3)."""
        title = paper.get("title", "")

        # Truncate if too long (keep under ~100K tokens)
        max_chars = 200000
        if len(full_text) > max_chars:
            half = max_chars // 2
            full_text = full_text[:half] + "\n\n[...TRUNCATED...]\n\n" + full_text[-half:]

        text = f"Title: {title}\n\n{full_text}"
        return self._extract_all_fields_combined(text, source="Full Text")

    def _extract_all_fields_combined(self, text: str, source: str) -> dict:
        """Extract ALL fields in a single LLM call (optimized)."""
        result = {
            "data_source": source,
            "extraction_confidence": "High" if source == "Full Text" else "Medium",
        }

        # Single combined prompt for all fields
        prompt = f"""Paper text:
{text[:60000]}

{COMBINED_EXTRACTION_PROMPT}"""

        try:
            response = get_completion(prompt, model=self.model)
            extracted = self._parse_json_response(response)
            result.update(extracted)
        except Exception as e:
            print(f"  Warning: Extraction failed: {e}")
            result.update(self._empty_all_fields())

        return result

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON response from LLM."""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Find the JSON content between ``` markers
            start = 1 if lines[0].startswith("```") else 0
            end = -1 if lines[-1] == "```" else len(lines)
            response = "\n".join(lines[start:end])

        try:
            data = json.loads(response)
            # Ensure all expected fields exist
            for field in ALL_FIELDS:
                if field not in data:
                    data[field] = "Not reported"
            return data
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return self._empty_all_fields()

    def _empty_extraction(self, reason: str) -> dict:
        """Return empty extraction with reason."""
        result = {
            "data_source": "None",
            "extraction_confidence": "None",
            "extraction_note": reason,
        }
        result.update(self._empty_all_fields())
        return result

    def _empty_all_fields(self) -> dict:
        """Return dict with all fields set to default values."""
        defaults = {}
        for field in ALL_FIELDS:
            if field.startswith("addresses_") or field.startswith("identifies_") or \
               field.startswith("handles_") or field.startswith("improves_"):
                defaults[field] = "N/A"
            elif field == "relevance_to_blkr201":
                defaults[field] = "Requires manual review"
            else:
                defaults[field] = "Not reported"
        return defaults


def batch_extract(
    papers: list,
    full_texts: dict = None,
    max_workers: int = 10,  # Increased default workers
) -> list:
    """Extract data from multiple papers in parallel.

    OPTIMIZED:
    - Uses single LLM call per paper (was 3)
    - Increased default workers to 10
    - Better progress reporting
    """
    full_texts = full_texts or {}
    extractor = NPDRExtractor()

    print(f"\nExtracting structured data from {len(papers)} papers...")
    print(f"Model: {EXTRACTION_MODEL}, parallel workers: {max_workers}")
    print(f"Optimization: Single LLM call per paper (3x faster)")

    def extract_paper(paper):
        pmid = str(paper.get("pmid", ""))
        full_text = full_texts.get(pmid, {})

        if full_text and full_text.get("full_text"):
            extracted = extractor.extract_from_full_text(paper, full_text["full_text"])
            if full_text.get("is_preprint"):
                extracted["data_source"] += " (Preprint)"
                extracted["preprint_note"] = full_text.get("note", "")
        else:
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
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"  Warning: Paper extraction failed: {e}")
                # Add original paper with empty extraction
                paper = futures[future]
                result = paper.copy()
                result.update(NPDRExtractor()._empty_extraction(str(e)))
                results.append(result)

            if completed % 5 == 0 or completed == len(papers):
                print(f"  Progress: {completed}/{len(papers)} papers extracted")

    print(f"Extraction complete!")
    return results
