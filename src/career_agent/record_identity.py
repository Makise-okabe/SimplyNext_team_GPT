import hashlib


def record_key(job: dict) -> str:
    if job.get("record_id"):
        return str(job["record_id"])
    identity = "|".join(" ".join(str(job.get(key) or "").lower().split()) for key in (
        "company", "title", "opportunity_type", "location", "job_id", "talentconnect_id",
    ))
    return hashlib.sha256(identity.encode()).hexdigest()[:20]
