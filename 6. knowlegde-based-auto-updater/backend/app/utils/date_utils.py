from datetime import datetime, timedelta
from typing import Optional

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string in ISO format"""
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None

def get_date_range(days: int) -> tuple[datetime, datetime]:
    """Get date range from now to specified days in the past"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date
