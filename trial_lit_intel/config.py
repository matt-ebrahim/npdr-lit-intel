"""Configuration management for Trial-Lit-Intel."""

import os
from pathlib import Path
from dotenv import load_dotenv


def get_secret(key: str, default: str = None):
    """Get a secret from Streamlit secrets or environment variables.

    Supports both local development (.env) and Streamlit Cloud (st.secrets).
    """
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fall back to environment variables
    return os.getenv(key, default)


def load_config() -> dict:
    """Load configuration from Streamlit secrets or environment variables."""
    # Load .env file from project root (for local development)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    config = {
        "litellm_api_key": get_secret("LITELLM_API_KEY"),
        "litellm_base_url": get_secret("LITELLM_BASE_URL"),
        "litellm_model": get_secret("LITELLM_MODEL", "opus-4-5"),
        "ncbi_api_key": get_secret("NCBI_API_KEY", ""),
    }

    # Validate required settings
    if not config["litellm_api_key"]:
        raise ValueError("LITELLM_API_KEY is required (set in .env or Streamlit secrets)")
    if not config["litellm_base_url"]:
        raise ValueError("LITELLM_BASE_URL is required (set in .env or Streamlit secrets)")

    return config


def load_snowflake_config() -> dict:
    """Load Snowflake configuration."""
    # Load .env file
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    return {
        "account": get_secret("SNOWFLAKE_ACCOUNT"),
        "user": get_secret("SNOWFLAKE_USER"),
        "warehouse": get_secret("SNOWFLAKE_WAREHOUSE"),
        "database": get_secret("SNOWFLAKE_DATABASE"),
        "schema": get_secret("SNOWFLAKE_SCHEMA"),
        "role": get_secret("SNOWFLAKE_ROLE"),
        "password": get_secret("SNOWFLAKE_PASSWORD", ""),
    }
