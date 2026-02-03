"""LiteLLM client for Claude API calls."""

import os
import litellm
from .config import load_config, get_secret

# Available models (update based on your LiteLLM gateway)
MODELS = {
    "opus": "claude-opus-4-5",
    "sonnet": "claude-sonnet-4-5",
    "haiku": "claude-haiku-4-5",
}

# Models that use specific API keys
HAIKU_MODELS = {"haiku", "claude-haiku-4-5", "claude-3-5-haiku", "claude-3-haiku"}
SONNET_MODELS = {"sonnet", "claude-sonnet-4-5", "claude-sonnet-4", "claude-3-5-sonnet"}


def get_completion(prompt: str, system_prompt: str = None, model: str = None) -> str:
    """Get a completion from Claude via LiteLLM.

    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt for context
        model: Model to use - "opus", "sonnet", "haiku", or full model name
               Defaults to config's litellm_model

    Returns:
        The model's response text
    """
    config = load_config()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Resolve model name
    if model is None:
        model_id = config['litellm_model']
    elif model in MODELS:
        model_id = MODELS[model]
    else:
        model_id = model

    # Use openai/ prefix for LiteLLM proxy (OpenAI-compatible API)
    model_name = f"openai/{model_id}"

    # Select API key based on model (each tier has separate key)
    if model in HAIKU_MODELS or model_id in HAIKU_MODELS:
        api_key = get_secret("LITELLM_HAIKU_API_KEY") or config["litellm_api_key"]
    elif model in SONNET_MODELS or model_id in SONNET_MODELS:
        api_key = get_secret("LITELLM_SONNET_API_KEY") or config["litellm_api_key"]
    else:
        api_key = config["litellm_api_key"]

    response = litellm.completion(
        model=model_name,
        messages=messages,
        api_key=api_key,
        api_base=config["litellm_base_url"],
    )

    return response.choices[0].message.content


def get_completion_async(prompt: str, system_prompt: str = None, model: str = None):
    """Async version for parallel calls (returns a future-like object)."""
    # For now, just wraps sync version - can be upgraded to true async later
    return get_completion(prompt, system_prompt, model)
