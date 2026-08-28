# Job Identity Verification — V1

## Goal

V1 does **not** search the web. It converts every extracted opportunity into a compact, grounded `JobIdentity` that V2 can use to search for the exact same role rather than a merely similar role.

## Why this layer exists

A title such as `Data Scientist` is not unique. Same-job verification needs stronger identity signals:

1. explicit Job / Requisition / Posting / Reference IDs;
2. company, exact role title and location;
3. business unit / team / duration / cohort;
4. distinctive phrases copied from the JD;
5. direct URLs already present in the email.

## Cost and latency rules

- Full newsletter/PDF content stays in the `EmailMessage` state.
- Each opportunity gets a bounded context of at most 4,200 characters.
- Up to four opportunities are extracted in one LLM call.
- A batch is capped at 15,000 source characters.
- Job identifiers and URLs are extracted deterministically.
- Every LLM distinctive phrase is grounded back against the supplied source text; hallucinated phrases are discarded.
- V1 makes zero web-search calls.

## Output

`JobIdentity` contains:

- company / title / location / opportunity type;
- explicit identifiers;
- business unit / team;
- duration / start and end period;
- target cohort;
- distinctive JD phrases;
- direct URLs;
- identity strength (`strong`, `moderate`, `weak`);
- stable source fingerprint.

## Acceptance criteria

V1 is complete when:

- unit tests cover ID extraction, bounded context, hallucination grounding, batching and stable fingerprinting;
- the live McKinsey email produces one `JobIdentity` with company/title/location and distinctive phrases from the attached JD;
- a real email containing an explicit Job/Requisition ID preserves the exact identifier when available;
- no public web search is performed.

## Next: V2

V2 consumes `JobIdentity` and routes progressively:

`direct URL -> exact Job ID search -> metadata search -> distinctive-phrase search`

with hard search-round and candidate limits.
