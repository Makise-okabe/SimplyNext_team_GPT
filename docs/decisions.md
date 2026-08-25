# Architecture Decisions

## ADR-001 — Do not access TalentConnect directly
NUS email is used as a discovery signal. Public/official company career pages are preferred as the verification source.

## ADR-002 — Email first
Telegram is deliberately excluded from Prototype 1.

## ADR-003 — Single orchestrated workflow first
Use LangGraph nodes plus one tool-using research loop before introducing multiple sub-agents.

## ADR-004 — No secrets in Git
OAuth tokens, `.env`, real emails, resumes and transcripts stay local.

## ADR-005 — Separate Signal from Job
An email mention is an `OpportunitySignal`; only after resolution/verification does it become a canonical `Job`.
