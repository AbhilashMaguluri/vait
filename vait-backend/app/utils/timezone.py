"""Centralized Timezone & Temporal Anchoring Utilities for VAIT.

Enforces Asia/Kolkata (IST, UTC+05:30) project-wide for backend logging,
evidence timestamps, relative date calculations, and LLM prompt grounding.
"""

import zoneinfo
from datetime import date, datetime, timedelta, timezone
from typing import Optional
import logging

from app.utils.config import get_settings

logger = logging.getLogger("vait.timezone")

# Standard IST fixed offset fallback in case zoneinfo data is missing on Windows/Alpine
IST_FIXED_OFFSET = timezone(timedelta(hours=5, minutes=30), name="IST")


def get_app_timezone() -> timezone:
    """Return the configured application timezone (defaults to Asia/Kolkata / IST)."""
    tz_name = "Asia/Kolkata"
    try:
        settings = get_settings()
        tz_name = getattr(settings, "app_timezone", "Asia/Kolkata") or "Asia/Kolkata"
        return zoneinfo.ZoneInfo(tz_name)
    except Exception as exc:
        logger.warning("Could not load ZoneInfo(%s): %s. Falling back to fixed IST offset.", tz_name, exc)
        return IST_FIXED_OFFSET


def now_ist() -> datetime:
    """Return the current timezone-aware datetime in IST."""
    return datetime.now(get_app_timezone())


def to_ist(dt: datetime) -> datetime:
    """Convert any datetime (aware or naive UTC) to IST."""
    tz = get_app_timezone()
    if dt.tzinfo is None:
        # Assume naive datetime is in UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz)


def format_ist(dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S IST") -> str:
    """Format a datetime as an IST string."""
    if dt is None:
        dt = now_ist()
    else:
        dt = to_ist(dt)
    return dt.strftime(fmt)


def resolve_relative_date(expression: str, reference_dt: Optional[datetime] = None) -> Optional[date]:
    """Resolve common relative date expressions against IST reference date.

    Supports:
    - 'today'
    - 'tomorrow'
    - 'yesterday'
    - 'day after tomorrow'
    - 'day before yesterday'
    - Weekdays: 'this monday', 'next friday', etc.
    """
    if not expression:
        return None

    if reference_dt is None:
        reference_dt = now_ist()
    else:
        reference_dt = to_ist(reference_dt)

    base_date = reference_dt.date()
    normalized = expression.strip().lower()

    if normalized in {"today", "todays", "today's"}:
        return base_date
    if normalized in {"tomorrow", "tomorrows", "tomorrow's"}:
        return base_date + timedelta(days=1)
    if normalized in {"yesterday", "yesterdays", "yesterday's"}:
        return base_date - timedelta(days=1)
    if normalized in {"day after tomorrow"}:
        return base_date + timedelta(days=2)
    if normalized in {"day before yesterday"}:
        return base_date - timedelta(days=2)

    weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    words = normalized.split()
    for word in words:
        if word in weekday_map:
            target_weekday = weekday_map[word]
            current_weekday = base_date.weekday()
            days_ahead = (target_weekday - current_weekday) % 7
            if "next" in words and days_ahead == 0:
                days_ahead = 7
            elif "next" in words and days_ahead > 0:
                days_ahead += 7
            return base_date + timedelta(days=days_ahead)

    return None


def get_ist_grounding_context() -> str:
    """Return a prompt grounding string with current IST date/time and institutional academic context."""
    curr = now_ist()
    day_name = curr.strftime("%A")
    date_str = curr.strftime("%d %B %Y")
    time_str = curr.strftime("%I:%M %p")
    iso_str = curr.isoformat()

    return (
        f"CURRENT TEMPORAL GROUNDING (Indian Standard Time - IST / Asia/Kolkata):\n"
        f"- Current Date & Time: {day_name}, {date_str}, {time_str} IST (ISO: {iso_str})\n"
        f"- Academic Context: Vasireddy Venkatadri Institute of Technology (VVIT) / VVIT University (VVITU), Guntur, Andhra Pradesh, India.\n"
        f"- All relative terms like \"today\", \"tomorrow\", \"yesterday\", \"current year\", \"this month\", or \"upcoming exams\" "
        f"MUST strictly be interpreted relative to this IST temporal anchor."
    )
