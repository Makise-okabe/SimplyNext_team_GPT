# SimplyNext Career Opportunity Agent

SimplyNext turns NUS career emails into a personalised, ranked stream of job opportunities.

A student uploads a resume and transcript. The agent builds a skills profile, reads forwarded Goh Ze Li / TalentConnect opportunities from the dedicated Outlook inbox, ranks every active role, resolves useful public job-page links, enriches job evidence when possible, and explains why each opportunity matches the student.

## Product flow

```text
Resume + Transcript
        |
Student skill/course profile
        |
Forwarded NUS career emails
        |
Deterministic extraction + canonical job catalog
        |
Rough ranking of every active email job
        |
Top 30 + exploration candidates
        |
Job Link Resolver
  official/ATS -> secondary -> unresolved
        |
Best-Effort JD Enricher
  full JD -> partial JD -> email/title evidence
        |
Re-rank + semantic Stage 2
        |
Top matches with skill evidence + clickable job page
        |
Related Job Discovery
  "You may also like" roles from top-matching companies
```

## Core design principles

- **Ranking must never depend on successful scraping.** Every active company/title can be ranked from trusted email evidence and title-derived role-family skills.
- **Job-page resolution is separate from JD extraction.** A useful LinkedIn/ATS/company job link is kept even when the page is dynamic or cannot yield a full JD.
- **Evidence is graded, not binary.** `full_jd`, `partial_jd`, and `source_only` are all valid matching inputs with different confidence.
- **Official sources are preferred, not mandatory.** Secondary pages may provide a clickable exact role or JD while provenance stays explicit.
- **No auto-apply.** The student remains the decision-maker.

## Main runner

```powershell
uv run python scripts/run_career_opportunity_agent.py `
  --resume "Du Yanzhang Resume.pdf" `
  --transcript "N_SR_TSRPT.pdf"
```

The output is written to:

```text
data/matching/career_opportunity_agent.json
```

It contains UI-ready `top_matches`, `related_jobs`, evidence levels, resolved page URLs, match explanations and summary metrics.

## Current ingestion

- Dedicated Outlook inbox via Microsoft Graph delegated `Mail.Read`
- Original forwarded sender recovery
- Goh Ze Li structured JOBS / INTERNSHIPS deterministic table extraction
- TalentConnect email/PDF extraction
- Canonical active job catalog and deduplication

## Matching

- Resume + transcript student profile
- NUS course-derived skills
- Deterministic broad Stage 1 ranking
- Best-effort web enrichment for promising roles only
- Batched semantic Stage 2 with rate-limit backoff and individual recovery
- Related-role discovery from high-ranking companies

## AWS deployment path

The local prototype is intentionally portable. The hosted hackathon version can map cleanly to:

- **S3** — resume/transcript uploads and generated artifacts
- **Lambda / API Gateway** — backend workflow endpoints
- **DynamoDB** — student profiles, canonical jobs and ranking state
- **Bedrock** — LLM extraction / semantic reasoning where appropriate
- **Amplify** — web UI hosting
- **CloudWatch** — logs and workflow diagnostics

AWS credentials and root secrets must never be committed to the repository.

## Architecture

See `docs/architecture.md`.

## Privacy

Never commit `.env`, OAuth tokens, raw mailbox exports, resumes/transcripts, or files from `private_data/`.
