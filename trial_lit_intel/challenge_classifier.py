"""LLM-based challenge classification for clinical AI literature.

This module allows users to describe their research needs in natural language
and maps them to standard clinical AI challenge categories.
"""

import json
from .llm_client import get_completion

# Standard Clinical AI Challenges (generalizable across indications)
STANDARD_CHALLENGES = {
    "long_term_prediction": {
        "name": "Long-term Prediction",
        "description": "Predicting patient outcomes 1+ years into the future",
        "examples": [
            "Predict disease progression over 2-5 years",
            "Forecast long-term treatment response",
            "Anticipate future complications",
        ],
        "extraction_field": "addresses_long_term_prediction",
    },
    "early_detection": {
        "name": "Early Detection / Early Signals",
        "description": "Identifying early biomarkers or signals before clinical manifestation",
        "examples": [
            "Detect subclinical disease",
            "Identify prodromal stages",
            "Find early warning signs before symptoms",
        ],
        "extraction_field": "identifies_early_signals",
    },
    "class_imbalance": {
        "name": "Class Imbalance / Low Event Rates",
        "description": "Handling rare events or imbalanced datasets in ML models",
        "examples": [
            "Predict rare adverse events",
            "Handle small positive class",
            "Address data scarcity for severe cases",
        ],
        "extraction_field": "handles_low_event_rates",
    },
    "rapid_progressors": {
        "name": "Rapid Progressors",
        "description": "Identifying patients who progress faster than typical",
        "examples": [
            "Find fast-progressing patients for trials",
            "Identify high-risk subgroups",
            "Predict accelerated disease course",
        ],
        "extraction_field": "identifies_rapid_progressors",
    },
    "diagnostic_consistency": {
        "name": "Diagnostic Consistency / Grading",
        "description": "Improving inter-reader agreement and diagnostic reproducibility",
        "examples": [
            "Reduce grader variability",
            "Standardize diagnostic criteria",
            "Automate consistent staging",
        ],
        "extraction_field": "improves_diagnostic_consistency",
    },
    "risk_stratification": {
        "name": "Risk Stratification",
        "description": "Stratifying patients by risk level for treatment decisions or trial enrollment",
        "examples": [
            "Identify high-risk vs low-risk patients",
            "Optimize patient selection for trials",
            "Personalize treatment based on risk",
        ],
        "extraction_field": "enables_risk_stratification",
    },
}

CHALLENGE_CLASSIFIER_PROMPT = """You are an expert at understanding clinical AI research needs.

The user will describe what they're looking for in clinical AI literature. Your task is to:
1. Understand their research intent
2. Map their needs to the standard clinical AI challenge categories below
3. Return which challenges are relevant

STANDARD CHALLENGES:
{challenges_description}

USER'S DESCRIPTION:
{user_description}

Return a JSON object with:
- "relevant_challenges": list of challenge IDs that match the user's needs
- "reasoning": brief explanation of why each challenge was selected
- "custom_focus": any specific focus areas mentioned that refine the challenges

Example output:
{{
    "relevant_challenges": ["long_term_prediction", "rapid_progressors"],
    "reasoning": "User wants to predict disease progression and identify fast progressors for clinical trials",
    "custom_focus": "Focus on 2-year prediction horizon for trial enrichment"
}}

Return ONLY valid JSON, no other text."""


def format_challenges_for_prompt() -> str:
    """Format challenge descriptions for the LLM prompt."""
    lines = []
    for cid, info in STANDARD_CHALLENGES.items():
        lines.append(f"- {cid}: {info['name']}")
        lines.append(f"  Description: {info['description']}")
        lines.append(f"  Examples: {', '.join(info['examples'])}")
        lines.append("")
    return "\n".join(lines)


def classify_user_intent(user_description: str) -> dict:
    """Classify user's research description into standard challenge categories.

    Args:
        user_description: Free-text description of what the user is looking for

    Returns:
        Dict with:
        - relevant_challenges: list of challenge IDs
        - reasoning: explanation
        - custom_focus: any specific refinements
        - challenge_details: full info for each relevant challenge
    """
    if not user_description.strip():
        # Default to all challenges if no description
        return {
            "relevant_challenges": list(STANDARD_CHALLENGES.keys()),
            "reasoning": "No specific description provided, using all standard challenges",
            "custom_focus": None,
            "challenge_details": STANDARD_CHALLENGES,
        }

    prompt = CHALLENGE_CLASSIFIER_PROMPT.format(
        challenges_description=format_challenges_for_prompt(),
        user_description=user_description,
    )

    try:
        response = get_completion(prompt, model="haiku")
        response = response.strip()

        # Parse JSON
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        result = json.loads(response)

        # Add full challenge details
        result["challenge_details"] = {
            cid: STANDARD_CHALLENGES[cid]
            for cid in result.get("relevant_challenges", [])
            if cid in STANDARD_CHALLENGES
        }

        return result

    except Exception as e:
        print(f"Warning: Challenge classification failed: {e}")
        # Return all challenges as fallback
        return {
            "relevant_challenges": list(STANDARD_CHALLENGES.keys()),
            "reasoning": f"Classification failed ({e}), using all challenges",
            "custom_focus": None,
            "challenge_details": STANDARD_CHALLENGES,
        }


def get_extraction_prompt_for_challenges(challenges: list[str], clinical_term: str) -> str:
    """Generate the extraction prompt section for the selected challenges.

    Args:
        challenges: List of challenge IDs to assess
        clinical_term: The clinical condition being studied

    Returns:
        Prompt text for challenge assessment
    """
    lines = ["## CHALLENGE ASSESSMENT (answer Y/N/Partial for each):"]

    for i, cid in enumerate(challenges, 1):
        if cid in STANDARD_CHALLENGES:
            info = STANDARD_CHALLENGES[cid]
            lines.append(f"{i}. {info['extraction_field']}: Does this paper address {info['name'].lower()}?")
            lines.append(f"   ({info['description']})")

    return "\n".join(lines)


def build_dynamic_extraction_prompt(
    challenges: list[str],
    clinical_term: str,
    custom_focus: str = None,
) -> str:
    """Build the full extraction prompt with dynamic challenges.

    Args:
        challenges: List of challenge IDs to assess
        clinical_term: The clinical condition
        custom_focus: Optional specific focus from user

    Returns:
        Complete extraction prompt
    """
    # Build challenge fields list
    challenge_fields = []
    for cid in challenges:
        if cid in STANDARD_CHALLENGES:
            info = STANDARD_CHALLENGES[cid]
            challenge_fields.append(info["extraction_field"])

    focus_text = f"\nSPECIFIC FOCUS: {custom_focus}" if custom_focus else ""

    prompt = f"""You are an expert at extracting structured information from scientific papers about AI/ML applications in {clinical_term}.
{focus_text}

Extract ALL of the following fields from the paper. Return a single JSON object.
If information is not available, use "Not reported" or "N/A".

## METHODOLOGY FIELDS:
1. task_type: Main ML task (e.g., "Progression Prediction", "Risk Stratification", "Grading")
2. imaging_modality: Imaging type if applicable (e.g., "CT", "MRI", "Fundus", "X-ray") or "N/A"
3. other_input_features: Non-imaging features (e.g., "Age, gender, lab values" or "None")
4. prediction_target: What model predicts (e.g., "Disease progression", "Treatment response")
5. prediction_horizon: Time frame (e.g., "1-5 years", "12 months", "N/A for cross-sectional")
6. model_architecture: ML/DL architecture (e.g., "ResNet-50", "XGBoost", "Transformer")
7. training_dataset: Dataset name/source
8. dataset_size: Sample size (e.g., "10,000 patients", "50,000 images")
9. external_validation: Validation details (e.g., "Yes (3 external sites)" or "No (internal CV only)")
10. primary_metric: Main metric (e.g., "AUC", "C-index", "Accuracy")
11. primary_metric_value: Metric value (e.g., "0.85", "0.79 ± 0.05")
12. secondary_metrics: Other reported metrics

## CHALLENGE ASSESSMENT (answer Y/N/Partial for each):
"""

    # Add dynamic challenges
    for i, cid in enumerate(challenges, 13):
        if cid in STANDARD_CHALLENGES:
            info = STANDARD_CHALLENGES[cid]
            prompt += f"{i}. {info['extraction_field']}: {info['description']}?\n"

    prompt += f"""
## SYNTHESIS:
{13 + len(challenges)}. relevance_to_study: "High"/"Medium"/"Low" - relevance to {clinical_term} clinical applications
{14 + len(challenges)}. potential_application: How findings could be applied (e.g., "Patient enrichment", "Endpoint optimization")
{15 + len(challenges)}. key_findings: 1-2 sentence summary of main findings
{16 + len(challenges)}. limitations: Main study limitations

Return ONLY a JSON object with all fields."""

    return prompt


# Convenience function to get challenge names for display
def get_challenge_display_names(challenge_ids: list[str]) -> list[tuple[str, str]]:
    """Get display names for challenge IDs.

    Returns:
        List of (id, display_name) tuples
    """
    return [
        (cid, STANDARD_CHALLENGES[cid]["name"])
        for cid in challenge_ids
        if cid in STANDARD_CHALLENGES
    ]
