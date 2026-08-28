# Job Identity Verification — V2 Progressive Candidate Discovery

## Goal

V2 consumes a V1 `JobIdentity` and discovers a **small ranked set of candidate web pages** that V3 can compare against the source JD.

V2 does **not** declare a job verified.

## Progressive routing

Search uses the strongest available evidence first:

1. direct URLs already supplied by the source email;
2. exact Job / Requisition / Posting / Reference ID search;
3. company + exact title + location (+ business unit when available);
4. company + distinctive JD phrases (+ location).

## Hard budgets

- maximum web-search rounds: **3**;
- maximum search results per query: **5**;
- maximum candidates handed to V3: **5**;
- V2 LLM calls: **0**;
- a strong exact-identifier candidate can stop later search rounds;
- a direct ATS/employer-like candidate can skip search entirely.

## Candidate evidence

Each `SearchCandidate` records:

- URL / host / result title / snippet;
- discovery strategy or strategies;
- explicit identifier hits;
- company/title/location/business-unit metadata hits;
- distinctive phrase hits;
- URL kind (`employer_or_ats`, `application_form`, `source_page`, `aggregator`, `unknown`);
- a deterministic **discovery score** used only for ranking candidates.

The discovery score is **not a verification confidence** and must never be shown as "same job probability".

## Stop conditions

V2 stops when one of the following occurs:

- direct employer/ATS candidate is already available;
- a strong exact-identifier candidate is found;
- all progressive queries are completed;
- the three-search hard budget is reached.

## Metrics

The V2 runner reports:

- web search calls;
- raw results seen;
- unique candidates;
- final candidates;
- V2 LLM calls (always 0);
- search latency;
- stop reason.

## Acceptance criteria

V2 is complete when:

- unit tests cover query order, URL classification, exact-ID scoring, distinctive phrase evidence and candidate merging;
- McKinsey V1 identity produces bounded candidate discovery with <=3 search calls and <=5 final candidates;
- a synthetic/real identity containing a Job ID searches that ID first;
- V2 never labels a candidate verified.

## Next: V3

V3 will fetch the top candidate pages (preferably in parallel), compare source identity vs candidate evidence, detect contradictions, and produce:

- `same_role`;
- `ambiguous`;
- `reject`;

with an optional single LLM adjudication only for genuinely ambiguous cases.
