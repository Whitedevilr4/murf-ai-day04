from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("wellness_log")

# Save wellness_log.json next to this file
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "wellness_log.json"


@dataclass
class WellnessEntry:
    timestamp: str
    mood: str
    energy: str
    stresses: str
    goals: List[str]
    self_care: List[str]
    summary: str


def _load_all_entries() -> List[Dict[str, Any]]:
    """Load all log entries from the JSON file. Returns an empty list if file does not exist."""
    if not LOG_PATH.exists():
        logger.info("No existing wellness log at %s, returning empty list", LOG_PATH)
        return []

    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from %s: %s", LOG_PATH, e)
        return []

    if isinstance(data, list):
        logger.info("Loaded %d existing entries from %s", len(data), LOG_PATH)
        return data
    else:
        logger.warning("Unexpected JSON structure in %s, resetting to empty list", LOG_PATH)
        return []


def append_entry(entry: WellnessEntry) -> str:
    """
    Append a new wellness entry to the JSON log.
    Creates wellness_log.json if it does not exist.
    """
    logger.info("Appending new wellness entry to %s", LOG_PATH)
    entries = _load_all_entries()
    entries.append(asdict(entry))

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    logger.info("Successfully wrote %d entries to %s", len(entries), LOG_PATH)
    return str(LOG_PATH)


def load_last_entry() -> Optional[Dict[str, Any]]:
    """
    Load the last logged wellness entry, or None if there is no log yet.
    """
    entries = _load_all_entries()
    if not entries:
        logger.info("No entries found in wellness log")
        return None

    last = entries[-1]
    logger.info("Loaded last entry with timestamp=%s", last.get("timestamp"))
    return last
