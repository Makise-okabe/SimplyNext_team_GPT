# Architecture — SimplyNext Prototype

## Goal

Convert forwarded NUS career information into evidence-backed, deduplicated opportunity records without requiring direct access to a student's institutional mailbox.

## Live flow

```text
NUS career email
      |
student forwarding rule
      |
dedicated SimplyNext Outlook inbox
      |
Microsoft Graph (delegated Mail.Read)
      |
recover original sender + subject
      |
parse HTML / links / PDF attachments in memory
      |
normalized EmailMessage
      |
LangGraph Track B
      |
OpportunitySignal
      |
public / official web research
      |
verification
      |-------------------------------|
      |                               |
official web matched          trusted NUS attachment matched
      |                               |
verified                     source_verified
      |                               |
      |----------- structured record -|
                      |
             optional SQLite memory
             (no raw email/PDF text)
```

## Privacy-minimised ingestion

The dedicated inbox is an ingestion gateway, not an archive copied into the repository. PDF bytes are fetched from Graph and parsed in memory. Raw email bodies, PDF bytes and attachment text are not persisted by default.

When product memory is enabled, SimplyNext stores only normalized opportunity fields such as company, title, location, deadline, verification state and source provenance. This lets the prototype deduplicate opportunities and support a future dashboard without retaining raw mailbox content.

## Verification semantics

- `verified`: identity matched against a fetched public/official posting.
- `source_verified`: identity matched directly against a PDF attachment from a trusted NUS career source, but no live official posting was proven.
- `partial`: a public URL exists but evidence is incomplete or inaccessible.
- `unresolved`: neither public evidence nor trusted attachment evidence is sufficient.

This distinction avoids pretending that an old or removed official posting is still live simply because NUS distributed a valid JD.

## Design principle

LangGraph is the orchestrator.

Use deterministic Python for sender recovery, HTML/PDF parsing, URL extraction, attachment handling, deduplication, verification checks and storage. Use LLM reasoning only where ambiguity exists: extracting opportunity signals, researching candidate postings and interpreting free-form requirements.

## Next product layers

The current `OpportunityStore` is intentionally replaceable. A later hosted version can swap SQLite for DynamoDB/PostgreSQL without changing the Graph ingestion or LangGraph workflow. The structured store can feed a student profile matcher, ranking layer, notification agent and web UI.
