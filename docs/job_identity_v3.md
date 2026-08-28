# Job Identity Verification — V3

## Goal

V3 answers a stricter question than search relevance:

> Does this fetched public page represent the **same employment opportunity** described by the source email/JD?

A high V2 search rank is never treated as verification by itself.

## Evidence hierarchy

Strongest to weaker:

1. exact Job / Requisition / Posting / Reference ID;
2. direct employer/ATS URL plus matching job identity;
3. company + title + business unit/team + distinctive JD wording;
4. company + title + location only (insufficient by itself for `verified`).

Explicit same-kind identifier conflicts are hard rejection evidence.

## Candidate handling

- Outlook navigation links and `aka.ms` transport links are removed before V3.
- Microsoft/Google/Typeform application forms are preserved as `application_url`, never as official identity proof.
- Up to four non-form candidates are fetched concurrently.
- Fetched page evidence is scored deterministically.
- One bounded GPT-OSS-120B tie-breaker call is allowed only when deterministic evidence leaves genuinely possible candidates.
- The LLM cannot elevate a weak candidate: it must already have company/title support plus an ID, business-unit match, or distinctive-JD match.

## Decisions

- `verified`: a fetched public candidate is proven to represent the same job.
- `source_verified`: the trusted NUS attachment strongly identifies the role but no public page proved same-job identity.
- `ambiguous`: multiple plausible candidates cannot be distinguished safely.
- `unresolved`: evidence is insufficient.

## V2 cleanup included with V3

The McKinsey live test exposed two V2 issues:

1. `outlook.live.com` and `aka.ms` were entering candidate lists as direct URLs;
2. the search query over-quoted both full title and business unit, returning zero results.

V3 includes a V2 cleanup: transport URLs are removed and progressive queries are relaxed to use a named business unit first, then an independent distinctive JD phrase.

## Cost / latency budget

- V1: normally one bounded identity-extraction LLM call.
- V2: max three search calls, zero LLM calls.
- V3: max four candidate fetches, fetched concurrently.
- V3 LLM judge: zero in clear cases; max one in ambiguous cases.

Liveness/open-vs-closed refresh and persistent verification cache remain V4 responsibilities.
