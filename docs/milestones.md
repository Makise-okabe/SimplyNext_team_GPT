# Milestones

## M0 — Repository scaffold
Status: implemented.

Definition of done:
- local project opens in VS Code
- uv environment works
- Git initialized
- GitHub remote created

## M0.5 — Outlook feasibility
Status: Graph Explorer passed; custom app waits for NUS admin consent.

Definition of done:
- NUS account can or cannot consent to/read mail through Microsoft Graph
- result documented
- no password stored

## M1 — Email parser / normalizer
Status: implemented and tested on real exported Graph email JSON.

Definition of done:
- parse sender, subject, date, HTML/text body, hyperlinks, attachment metadata
- normalize HTML and plain-text URLs

## M2 — Career filter
Status: implemented.

Definition of done:
- known career sources and subject patterns are identified without an LLM

## M3 — Opportunity signal extraction
Status: implemented; real LLM smoke test requires local GROQ_API_KEY.

Definition of done:
- email -> structured company/role/type/deadline/location/source/links
- large newsletters are split into bounded candidate chunks before LLM use

## M4 — Link resolver
Status: implemented.

Definition of done:
- public links and redirects are resolved
- page title/text evidence is captured
- login walls and failed pages remain explicit rather than fabricated

## M5 — Job research agent
Status: implemented.

Definition of done:
- opportunity signal -> direct public-page research when possible
- fallback public web search when email links are incomplete/login-walled
- likely employer/job pages are supplied as evidence to the LLM

## M6 — Verification
Status: implemented conservatively.

Definition of done:
- canonical Job contains source/evidence
- verified / partial / unresolved status is based on fetched evidence
- TalentConnect/login-wall-only evidence is never treated as fully verified

## M7 — LangGraph end-to-end
Status: implemented; final real-data smoke run requires local GROQ_API_KEY.

Definition of done:
- one real NUS career email runs through:
  filter -> normalize -> signal extraction -> resolve/search -> research -> verify
- final output can be saved to data/track_b_results.json

Run locally:

```powershell
git checkout feature/track-b-end-to-end
git pull
uv run pytest
uv run python scripts/run_track_b.py --limit 1
```

## M8 — Live Microsoft Graph connector
Status: Track A, blocked on NUS admin consent; intentionally isolated from Track B.

Definition of done:
- Graph inbox input can replace local exported Graph JSON without changing downstream graph
