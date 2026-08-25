# NUS Career Agent

Agentic AI prototype for turning NUS career emails into structured, verifiable job opportunities.

The project focuses on a narrow workflow rather than a generic career-advice chatbot: ingest trusted NUS career signals, extract useful opportunity data, research the public/official posting, verify key facts, and produce a canonical job record that can later be matched against a student's profile.

## Prototype 1

```text
NUS career email
    ↓
Microsoft Graph / local EML connector
    ↓
Trusted-source + career-email filter
    ↓
Normalize message content and links
    ↓
Extract opportunity signal
    ↓
Resolve links / research official posting
    ↓
Verify role, deadline, location and eligibility evidence
    ↓
Canonical job record
```

Prototype 1 deliberately starts with two NUS career-email sources:

- NUS TalentConnect notifications
- ECE career-adviser emails

The goal is to prove the end-to-end workflow before expanding to other sources such as company career pages, LinkedIn, Tech in Asia or Telegram channels.

## Current status

- [x] Repository scaffold and `uv` environment
- [x] Microsoft Graph delegated `Mail.Read` connector
- [x] Device-code authentication with local token cache
- [x] Trusted sender targeting for the two Prototype 1 sources
- [x] Deterministic career-email filter
- [x] Local EML connector scaffold
- [x] Unit test for career-email filtering
- [ ] Structured opportunity-signal extraction
- [ ] Link resolver
- [ ] Official job research step
- [ ] Evidence-backed verification
- [ ] Canonical job storage
- [ ] End-to-end LangGraph workflow
- [ ] Student profile / resume matching

## Design principles

- **Narrow workflow first.** The agent should solve one concrete job-search problem well instead of acting like a generic career adviser.
- **Deterministic where possible.** Filtering, parsing, redirects, dates, deduplication and storage should use normal Python logic.
- **LLM only for ambiguity.** Use reasoning for free-form extraction, search planning and matching a career signal to the correct official posting.
- **Evidence over guessing.** Deadlines, eligibility and visa-related facts should ultimately be tied back to a source.
- **Human keeps the decision.** The system surfaces and verifies opportunities; it does not auto-apply or decide a career path for the user.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the current workflow and [`docs/milestones.md`](docs/milestones.md) for the build sequence.

## Development

Install dependencies and run tests with:

```bash
uv sync
uv run pytest
```

Microsoft Graph credentials belong in a local `.env` file based on `.env.example`.

## Privacy and security

Never commit:

- `.env`
- OAuth access or refresh tokens
- `token_cache.json`
- real `.eml` files
- resumes or transcripts
- personal mailbox exports
- anything in `private_data/`

The Graph connector uses delegated `Mail.Read` access. Password entry happens on Microsoft's login page, not inside this project.
