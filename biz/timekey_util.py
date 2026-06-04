"""RULE_TIMEKEY / START_TIME / END_TIME 문자열 연산."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


def parse_timekey(value: str) -> Optional[datetime]:
    """숫자 14자리 이상이면 YYYYMMDDHHMMSS 로 파싱."""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 14:
        return None
    try:
        return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def format_timekey(dt: datetime) -> str:
    """16자리 규격 (초 뒤 00 패딩)."""
    return dt.strftime("%Y%m%d%H%M%S") + "00"


def add_hours_to_timekey(base: str, hours: float) -> str:
    dt = parse_timekey(base)
    if dt is None:
        return base
    return format_timekey(dt + timedelta(hours=hours))


def resolve_horizon_start(rule_timekey: str, fallback: Optional[str] = None) -> str:
    """간트/스케줄 구간 시작 시각 문자열."""
    if parse_timekey(rule_timekey) is not None:
        return rule_timekey
    if fallback and parse_timekey(fallback) is not None:
        return fallback
    return rule_timekey
