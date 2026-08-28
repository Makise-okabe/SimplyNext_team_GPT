# Job Identity Verification — V4

## Goal

V4 keeps V2/V3's conservative same-job logic unchanged and adds the operational layer needed for a usable prototype:

1. structured cache;
2. independent liveness refresh;
3. latency/call-budget metrics.

## Cache hierarchy

### Source identity cache

Key: Outlook `message_id` + cache schema version.

Stores only compact `JobIdentity` objects. Raw email body, PDF text, attachment bytes and `evidence_snippets` are not persisted. A warm hit skips both opportunity-signal extraction and V1 identity extraction.

### Same-job verification cache

The cache key is cycle-aware:

`source_fingerprint + source email year + start period + end period + duration`

This prevents a recurring 2027 internship from blindly reusing a 2026 verification result.

TTL is status-aware:

- `verified`: 30 days;
- `source_verified`: 7 days;
- `ambiguous`: 1 day;
- `unresolved`: 1 day.

Identity and liveness are intentionally cached separately.

### Liveness cache

Key: verified official URL.

TTL: 6 hours.

A liveness refresh never re-runs web search or the same-job LLM judge. It only re-checks the already-verified URL.

## Conservative liveness semantics

- `closed`: HTTP 404/410 or explicit expired/no-longer-available wording;
- `open`: HTTP 2xx plus a strong application marker such as `Apply now`;
- `unknown`: reachable page without reliable open/closed evidence, access problems, or generic careers-page redirect.

A plain HTTP 200 is **not** automatically called open.

## Warm-cache target

For the same immutable Outlook message and a still-fresh V3 result:

- signal LLM calls: 0;
- identity LLM calls: 0;
- V2 search calls: 0;
- V3 candidate fetches: 0;
- V3 judge LLM calls: 0.

If the role has no verified official URL (for example a `source_verified` NUS-only opportunity), liveness also performs 0 fetches.

If an official URL exists but liveness TTL expired, only one cheap liveness fetch is required.

## Metrics

V4 reports cache hits/misses, LLM calls, search calls, V3 fetch calls, liveness fetches, processing latency and approximate LLM source characters. The character count is explicitly not presented as billed-token usage because provider token accounting is not yet wired into the structured-output calls.

## Live acceptance test

Use a fresh V4 cache and process the same McKinsey message twice:

```powershell
uv run python scripts/run_job_identity_v4.py --scan 10 --limit 1 --subject "McKinsey" --reset-cache --repeat 2
```

Expected behaviour:

- pass 1: cache misses and normal V1/V2/V3 work;
- pass 2: source + verification cache hits, zero LLM/search/V3 fetch calls;
- McKinsey remains `source_verified` with the Microsoft Form as `application_url`;
- pipeline errors remain zero.
