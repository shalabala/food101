def time_string(secs: float) -> str:
    """Convert seconds to a string in the format of HH:MM:SS."""
    hours, remainder = divmod(int(secs), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"
    