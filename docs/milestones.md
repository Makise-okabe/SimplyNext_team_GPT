# Milestones

## M0 — Repository scaffold
Definition of done:
- local project opens in VS Code
- uv environment works
- Git initialized
- GitHub remote created

## M0.5 — Outlook feasibility
Definition of done:
- NUS account can or cannot consent to/read mail through Microsoft Graph
- result documented
- no password stored

## M1 — EML parser
Definition of done:
- parse sender, subject, date, HTML/text body, hyperlinks, attachment metadata

## M2 — Career filter
Definition of done:
- known career sources and subject patterns are identified without an LLM

## M3 — Opportunity signal extraction
Definition of done:
- email -> structured company/role/type/deadline/location/source/links

## M4 — Link resolver
Definition of done:
- "click here" style HTML links are extracted and public redirects resolved

## M5 — Job research agent
Definition of done:
- opportunity signal -> web research -> likely official company posting

## M6 — Verification
Definition of done:
- canonical job record contains source/evidence and verified/unresolved status

## M7 — LangGraph end-to-end
Definition of done:
- one real NUS career email runs through the entire pipeline

## M8 — Live Microsoft Graph connector
Definition of done:
- Graph inbox input can replace local .eml input without changing downstream graph
