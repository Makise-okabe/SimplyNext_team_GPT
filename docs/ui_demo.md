# SimplyNext UI and verified discovery

The UI scans trusted career emails from the dedicated Outlook inbox, builds a student profile once, and passes it to the career runner. The backend ranks the opportunity pool, researches a bounded shortlist, and exports result cards through a shared verification boundary.

## Run locally

```powershell
git switch feature/trusted-job-discovery
uv sync
uv run streamlit run ui/app.py
```

Existing Outlook authentication, Groq/search configuration, and NUSMods resources are required for a live run. Upload a resume and transcript PDF, then select **Scan Outlook & find matches**.

## Job links and evidence

A job button requires a fetched destination whose role, employer, and available identity fields match the email opportunity. Homepages, sign-in pages, wrong roles, closed listings, and insufficient page evidence do not become job buttons. An unverified opportunity stays available for evidence-limited matching.

JSON-LD JobPosting data is extracted before scripts are removed. Public Workday and Greenhouse detail resources are supported for recognized posting URLs. The job description and destination are taken from the same verified result. Ranking cannot restore stale application URLs or reopen a closed posting.

Research is limited to three search queries and six candidate fetches per role. Repeated queries and fetches are reused within a run. The UI does not run a separate link-rescue search or rebuild the student profile in the subprocess.

The results show a shortlist, a searchable opportunity pool, source information, fit evidence, gaps, and link status. Different known job IDs, TalentConnect IDs, or locations remain separate during consolidation and ranking lookup.

## Validation and remaining checks

Run `uv run pytest -q`. Regression coverage includes structured job data, misleading destinations, stale result aliases, closed postings, duplicate identities, and per-run fetch reuse. Streamlit AppTest can exercise landing/results rendering without mailbox access.

These deterministic checks do not measure live retrieval quality. Before the demo, run the real inbox workflow and manually inspect the returned destinations for the correct employer, role, location, and application availability. Record resolved links versus researched roles, JD coverage, and runtime. Browser visual review and live end-to-end verification remain necessary.
