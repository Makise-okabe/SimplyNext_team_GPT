# SimplyNext Hackathon UI

The UI is intentionally thin: it does not duplicate or change the frozen Career Opportunity Backend V2. It uploads a resume and transcript, invokes the existing one-command backend runner, then renders its JSON output as an evidence-backed career dashboard.

## Local quickstart

From the repository root:

```powershell
git switch feature/hackathon-ui
uv sync
uv run streamlit run ui/app.py
```

Streamlit will print a local URL, normally:

```text
http://localhost:8501
```

## Prerequisites

The same backend prerequisites still apply:

- `.env` contains the existing Groq/search configuration.
- `data/job_records/latest_matching_candidates.json` exists.
- The NUSMods database/cache used by the frozen backend is available locally.

The `data/` directory stays gitignored because it contains generated/local pipeline state.

To point the UI at a different generated job catalogue without changing code:

```powershell
$env:SIMPLYNEXT_JOBS_PATH="data/job_records/latest_matching_candidates.json"
uv run streamlit run ui/app.py
```

## Demo flow

1. Open the landing page.
2. Upload a resume PDF.
3. Upload a transcript PDF.
4. Click **Find my opportunities**.
5. The UI shows four backend phases as human-readable progress:
   - student profile construction,
   - rough ranking + targeted web enrichment,
   - semantic fit validation,
   - related-role discovery.
6. The results dashboard shows:
   - profile/course/skill counts,
   - backend pipeline metrics,
   - top match scores,
   - why-match explanations,
   - matched skills,
   - supporting vs missing evidence,
   - View Job / Company Careers / Search Job actions,
   - You May Also Like recommendations.

## Judge-facing product behavior

The CTA intentionally reflects link quality:

- **View Job**: sufficiently confident job-specific page.
- **Company Careers**: a real company/careers destination exists but is not treated as an exact posting.
- **Search Job**: no reliable exact URL; the UI falls back to a search instead of pretending a link is exact.

This keeps the product aligned with the backend's confidence model and prevents the UI from reintroducing the wrong-link problem that the backend regression tests were designed to catch.

## Architecture boundary

```text
Streamlit UI
    |
    | uploads PDFs
    v
scripts/run_career_opportunity_agent.py   (frozen backend contract)
    |
    v
data/matching/career_opportunity_agent.json
    |
    v
UI cards / evidence / links / related roles
```

The UI branch should not change matching weights, scraping logic, semantic prompts, extraction logic, or URL resolution rules.
