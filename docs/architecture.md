# Architecture — Prototype 1

## Goal

Convert NUS career-related Outlook emails into verified public/official job records.

## Core flow

```text
START
  |
load_email
  |
filter_email
  |
career related?
  |-- no --> END
  |
 yes
  |
normalize_email
  |
extract_signal
  |
enough information?
  |-- no --> resolve_links --> extract_signal
  |
research_job
  |
official posting found?
  |-- no --> unresolved
  |
 yes
  |
verify_job
  |
save_job
  |
END
```

## Design principle

LangGraph is the orchestrator.

Use deterministic Python for:
- sender/subject filtering
- HTML parsing
- URL extraction
- redirect handling
- deduplication
- date comparisons
- storage

Use LLM reasoning only where ambiguity exists:
- extracting opportunity signals from free-form career emails
- deciding what to search for
- matching a signal to a candidate official posting
- resolving ambiguous job requirements

## Future architecture

After the single workflow is stable, split only where context isolation is useful:

- Supervisor
- Email Intelligence Agent
- Job Research Agent
- Profile Match Agent

Future deployment may wrap the same LangGraph workflow in AWS Bedrock AgentCore.
