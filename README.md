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
Email-only rough ranking of every active job
        |
Small high-value shortlist
        |
AWS AgentCore Web Search + Job Link Resolver
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
- **No generated search buttons.** The results page shows only a concrete role URL returned by search and checked against the company/title; it never fabricates a Google or LinkedIn search page.
- **No auto-apply.** The student remains the decision-maker.

## Main runner

First authenticate through AWS IAM Identity Center. Do not copy temporary access
keys into source files or commit them to Git:

```powershell
aws configure sso --profile simplenext-hackathon
aws sso login --profile simplenext-hackathon
aws sts get-caller-identity --profile simplenext-hackathon
```

Use the SSO Start URL and SSO Region shown by the AWS Access Portal. The SSO
Region identifies IAM Identity Center; it can be `ap-southeast-1` while the
Bedrock/AgentCore service Region below is `us-east-1`.

Copy `.env.example` to `.env`, then set only the profile name and the non-secret
AgentCore values produced by the setup command:

```env
AWS_PROFILE=simplenext-hackathon
AWS_REGION=us-east-1
AWS_BEDROCK_REGION=us-east-1
AWS_BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

Verify Bedrock and create the managed AWS web-search gateway once:

```powershell
uv run python scripts/check_aws_access.py
uv run python scripts/setup_aws_agentcore_search.py
```

Add the printed `AWS_AGENTCORE_GATEWAY_URL` and
`AWS_AGENTCORE_WEB_SEARCH_TOOL` values to `.env`. Then run:

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
- Deterministic full-inbox prefilter before live research: 12 strongest roles
  plus 3 exploration roles by default
- Demo cost guard: at most 15 AgentCore network attempts per process, with
  successful search results reused from a 12-hour local cache

## AWS architecture

The active AI/search path is AWS-native:

- **Amazon Bedrock Converse** — Nova Lite extraction, job verification and semantic matching
- **Amazon Bedrock AgentCore Gateway + Web Search Tool** — live URL discovery without Tavily or Groq

The hosted hackathon version can additionally map to:

- **S3** — resume/transcript uploads and generated artifacts
- **Lambda / API Gateway** — backend workflow endpoints
- **DynamoDB** — student profiles, canonical jobs and ranking state
- **Amplify** — web UI hosting
- **CloudWatch** — logs and workflow diagnostics

Use Bedrock on-demand only. Avoid EC2, RDS, NAT Gateway, load balancers,
OpenSearch and provisioned throughput for the hackathon account. Monitor the
Innovation Sandbox budget because access can be revoked when the actual budget
limit is reached.

AWS credentials, SSO caches and root secrets must never be committed to the repository.

## Architecture

See `docs/architecture.md`.

## Privacy

Never commit `.env`, OAuth tokens, raw mailbox exports, resumes/transcripts, or files from `private_data/`.
