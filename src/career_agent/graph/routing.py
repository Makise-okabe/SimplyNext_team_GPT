def route_career_email(state: dict) -> str:
    return "normalize_email" if state.get("is_career_email") else "end"
