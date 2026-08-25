# NUS Career Agent

Agentic AI prototype for turning NUS career emails into verified job opportunities.

## Prototype 1

Email signal -> parse content/links -> extract opportunity signal -> resolve public link ->
research official company career site -> verify -> canonical job record.

## Current status

- [x] Repository scaffold
- [ ] Microsoft Graph Mail.Read feasibility test
- [ ] EML parser
- [ ] Career email filter
- [ ] Signal extraction
- [ ] Link resolver
- [ ] Job research agent
- [ ] Verification
- [ ] End-to-end LangGraph workflow

## Architecture

See `docs/architecture.md`.

## Privacy

Never commit:
- `.env`
- OAuth tokens
- real `.eml` files
- resumes/transcripts
- any file in `private_data/`
