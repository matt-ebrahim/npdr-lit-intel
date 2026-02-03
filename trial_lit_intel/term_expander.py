"""Clinical term expansion using Claude."""

import json
from .llm_client import get_completion


SYSTEM_PROMPT = """You are a medical terminology expert. Your task is to identify the most relevant search terms for finding AI/ML research literature on clinical/medical conditions.

For each term, select the TOP 4 most important related terms that would capture the most relevant literature. Prioritize:
1. The most commonly used abbreviation (if one exists)
2. The most important clinical subtypes that have significant AI/ML research
3. Key alternative names used in medical literature

Return your response as a JSON object with the following structure:
{
    "original": "the input term",
    "synonyms": ["exactly", "four", "most", "relevant"]
}

IMPORTANT:
- Return EXACTLY 4 synonyms (plus the original = 5 total terms)
- Only include clinically accurate and established medical terminology
- Focus on terms most likely to appear in AI/ML research papers
- Do not invent terms."""


def expand_clinical_term(term: str) -> list[str]:
    """Expand a clinical term into synonyms and related terms.

    Args:
        term: The clinical term to expand (e.g., "diabetic retinopathy")

    Returns:
        List of all terms including original and synonyms
    """
    prompt = f"""For the clinical term: "{term}"

Select the TOP 4 most relevant related terms for finding AI/ML research literature. Choose terms that would capture the most important papers in this field.

Return ONLY the JSON object with exactly 4 synonyms, no additional text."""

    response = get_completion(prompt, SYSTEM_PROMPT)

    # Parse JSON from response
    # Handle potential markdown code blocks
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(lines[1:-1])

    try:
        data = json.loads(response)
        all_terms = [data["original"]] + data["synonyms"]
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for t in all_terms:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique_terms.append(t)
        return unique_terms
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse LLM response as JSON: {e}")
        print(f"Response was: {response}")
        # Fallback: return just the original term
        return [term]
