"""Persistent file-based cache for expensive API results.

Caches full text and extraction results to disk to avoid redundant API calls
across runs. Uses a 30-day TTL by default.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any

# Default cache directory
CACHE_DIR = Path.home() / ".cache" / "trial-lit-intel"
CACHE_TTL_DAYS = 30


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(key: str, cache_type: str) -> Path:
    """Get cache file path for a key.

    Args:
        key: Cache key (e.g., PMID)
        cache_type: Type of cached data (e.g., "pmc", "extraction")

    Returns:
        Path to cache file
    """
    _ensure_cache_dir()
    # Sanitize key for filesystem
    safe_key = "".join(c if c.isalnum() else "_" for c in str(key))
    return CACHE_DIR / f"{cache_type}_{safe_key}.json"


def get_cached(key: str, cache_type: str) -> Optional[Any]:
    """Get cached result if exists and not expired.

    Args:
        key: Cache key
        cache_type: Type of cached data

    Returns:
        Cached result or None if not found/expired
    """
    path = _get_cache_path(key, cache_type)

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))

        # Check if expired
        if datetime.now() - cached_at > timedelta(days=CACHE_TTL_DAYS):
            path.unlink()  # Delete expired cache
            return None

        return data.get("result")

    except (json.JSONDecodeError, KeyError, ValueError):
        # Corrupted cache file, delete it
        path.unlink(missing_ok=True)
        return None


def set_cached(key: str, cache_type: str, result: Any):
    """Cache a result.

    Args:
        key: Cache key
        cache_type: Type of cached data
        result: Data to cache (must be JSON serializable)
    """
    path = _get_cache_path(key, cache_type)

    data = {
        "_cached_at": datetime.now().isoformat(),
        "_cache_type": cache_type,
        "result": result,
    }

    try:
        path.write_text(json.dumps(data, default=str))
    except (TypeError, OSError) as e:
        print(f"  Warning: Failed to cache {cache_type}/{key}: {e}")


def clear_cache(cache_type: str = None):
    """Clear cached data.

    Args:
        cache_type: If provided, only clear this type. Otherwise clear all.
    """
    _ensure_cache_dir()

    if cache_type:
        pattern = f"{cache_type}_*.json"
    else:
        pattern = "*.json"

    count = 0
    for path in CACHE_DIR.glob(pattern):
        path.unlink()
        count += 1

    print(f"Cleared {count} cached items")


def get_cache_stats() -> dict:
    """Get cache statistics.

    Returns:
        Dict with cache stats (count, size, types)
    """
    _ensure_cache_dir()

    stats = {
        "total_files": 0,
        "total_size_mb": 0,
        "by_type": {},
    }

    for path in CACHE_DIR.glob("*.json"):
        stats["total_files"] += 1
        stats["total_size_mb"] += path.stat().st_size / (1024 * 1024)

        # Extract type from filename
        cache_type = path.stem.split("_")[0]
        if cache_type not in stats["by_type"]:
            stats["by_type"][cache_type] = 0
        stats["by_type"][cache_type] += 1

    stats["total_size_mb"] = round(stats["total_size_mb"], 2)

    return stats
