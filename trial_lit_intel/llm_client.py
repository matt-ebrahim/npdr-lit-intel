"""LiteLLM client for Claude API calls with performance optimizations."""

import os
import asyncio
import litellm
from functools import lru_cache
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

# Cache config at module level (loaded once)
_config_cache = None


def _get_cached_config():
    """Get cached config (loads once per process)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def _get_api_key(model: str, model_id: str, config: dict) -> str:
    """Get the appropriate API key for the model."""
    if model in HAIKU_MODELS or model_id in HAIKU_MODELS:
        return get_secret("LITELLM_HAIKU_API_KEY") or config["litellm_api_key"]
    elif model in SONNET_MODELS or model_id in SONNET_MODELS:
        return get_secret("LITELLM_SONNET_API_KEY") or config["litellm_api_key"]
    return config["litellm_api_key"]


def _resolve_model(model: str, config: dict) -> str:
    """Resolve model shorthand to full model ID."""
    if model is None:
        return config['litellm_model']
    return MODELS.get(model, model)


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
    config = _get_cached_config()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    model_id = _resolve_model(model, config)
    model_name = f"openai/{model_id}"
    api_key = _get_api_key(model, model_id, config)

    response = litellm.completion(
        model=model_name,
        messages=messages,
        api_key=api_key,
        api_base=config["litellm_base_url"],
    )

    return response.choices[0].message.content


async def get_completion_async(prompt: str, system_prompt: str = None, model: str = None) -> str:
    """True async version using litellm.acompletion for parallel API calls."""
    config = _get_cached_config()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    model_id = _resolve_model(model, config)
    model_name = f"openai/{model_id}"
    api_key = _get_api_key(model, model_id, config)

    response = await litellm.acompletion(
        model=model_name,
        messages=messages,
        api_key=api_key,
        api_base=config["litellm_base_url"],
    )

    return response.choices[0].message.content


async def get_completions_parallel(prompts: list, system_prompt: str = None, model: str = None) -> list:
    """Run multiple completions in parallel using true async.

    Args:
        prompts: List of prompts to process
        system_prompt: Shared system prompt for all requests
        model: Model to use

    Returns:
        List of responses in same order as prompts
    """
    tasks = [
        get_completion_async(prompt, system_prompt, model)
        for prompt in prompts
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


def get_completions_parallel_sync(prompts: list, system_prompt: str = None, model: str = None) -> list:
    """Sync wrapper for parallel completions (for use in non-async code)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, use ThreadPoolExecutor fallback
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                futures = [
                    executor.submit(get_completion, prompt, system_prompt, model)
                    for prompt in prompts
                ]
                return [f.result() for f in futures]
        return loop.run_until_complete(get_completions_parallel(prompts, system_prompt, model))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(get_completions_parallel(prompts, system_prompt, model))
