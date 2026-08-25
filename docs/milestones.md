# Milestones

## M0 — Repository scaffold ✅
Definition of done:
- local project opens in VS Code
- `uv` environment works
- Git initialized
- GitHub remote created

## M0.5 — Outlook feasibility ✅
Definition of done:
- NUS account can read selected mail through Microsoft Graph
- delegated `Mail.Read` flow is documented in code
- no password is stored by the project
- token cache remains local and uncommitted

## M1 — Email ingestion 🟡
Definition of done:
- local EML connector can parse messages into the shared email model
- Microsoft Graph connector can retrieve selected mailbox messages
- both connectors feed the same downstream representation

Current state:
- Graph connector implemented
- EML connector scaffold present

## M2 — Career filter ✅
Definition of done:
- known career sources and subject patterns are identified without an LLM
- Prototype 1 limits ingestion to the intended trusted NUS career sources
- filter behavior has a unit test

## M3 — Opportunity signal extraction 🟡
Definition of done:
- email -> structured company / role / type / deadline / location / source / links
- one email may produce multiple opportunity signals
- ambiguous or missing fields are preserved rather than invented

Current state:
- source gate and node scaffold implemented
- structured extraction is still pending

## M4 — Link resolver
Definition of done:
- "click here" style HTML links are extracted
- public redirects are resolved
- tracking URLs can be mapped to useful destination URLs where possible

## M5 — Job research agent
Definition of done:
- opportunity signal -> web research -> likely official company posting
- official company career pages are preferred over secondary reposts
- research result carries source links and evidence

## M6 — Verification
Definition of done:
- canonical job record contains source/evidence
- role title, location, deadline and relevant eligibility facts are checked
- uncertain facts remain marked unresolved rather than guessed

## M7 — LangGraph end-to-end
Definition of done:
- one real NUS career email runs through the entire pipeline
- deterministic and LLM steps are orchestrated in one reproducible graph
- failures are captured without crashing the full workflow

## M8 — Profile matching
Definition of done:
- student profile can be built from resume/transcript information without committing private files
- job records can be compared with major, skills, GPA/eligibility constraints and region preferences
- output explains evidence and gaps instead of making the final career decision

## M9 — Additional sources
Definition of done:
- architecture can ingest additional job-search sources without rewriting the core job record pipeline
- possible sources include company graduate pages, LinkedIn, Tech in Asia and curated Telegram signals

## Product boundary

The project is an opportunity-discovery and verification assistant, not an auto-apply bot and not a generic career-advice chatbot. The human user retains the final decision.
