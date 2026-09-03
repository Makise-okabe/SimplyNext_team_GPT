# Architecture — SimplyNext Career Opportunity Agent v2

## Product goal

Convert forwarded NUS career opportunities into a personalised, ranked and clickable job feed after a student uploads a resume and transcript.

The prototype optimises for **useful coverage**, not perfect crawling. A role remains rankable even when a public page cannot be fetched.

## End-to-end flow

```text
NUS Goh / TalentConnect email
        |
Dedicated Outlook inbox
        |
Microsoft Graph Mail.Read
        |
Recover original sender + parse email/PDF
        |
Deterministic / bounded-LLM extraction
        |
Canonical active job catalog
        |
        |-------------------------------|
        |                               |
Resume + Transcript                     |
        |                               |
Student profile                         |
explicit + course-derived skills        |
        |                               |
        |------------ Stage 1 ----------|
                     |
             Rough rank all jobs
                     |
            Top 30 + exploration
                     |
              Job Link Resolver
                     |
      official exact / probable
      secondary exact / probable
      company careers / unresolved
                     |
           Best-Effort JD Enricher
                     |
      full JD / partial JD / source-only
                     |
                 Re-rank
                     |
          Batched semantic Stage 2
                     |
              Final Top Matches
                     |
       skill evidence + match reason
           + clickable job page
                     |
          Related Job Discovery
       from top-matching companies
                     |
              "You may also like"
```

## Why link resolution and JD extraction are separate

A job page can be useful to the student even when it is JavaScript-rendered, protected by anti-bot measures, or too sparse to produce a complete JD.

Therefore each `JobRecord` stores independent page state:

- `job_page_url`
- `job_page_kind`
- `job_page_confidence`

and independent JD state:

- `fetched_official`
- `fetched_secondary`
- `partial_official`
- `partial_secondary`
- `source_context_only`
- `unavailable`

A failed JD fetch never deletes a resolved job link.

## Matching evidence hierarchy

```text
full_jd
  strongest employer-role evidence

partial_jd
  useful fetched role evidence, but incomplete

source_only
  trusted NUS email + company/title + role-family inference
```

All three are valid ranking inputs. Evidence level affects confidence, not basic eligibility.

Title-derived skills are allowed for broad relevance, e.g. `Embedded Software Engineer` can imply embedded systems / C/C++ / digital design. They are never presented as employer-stated requirements.

## Web strategy

Web work is concentrated on promising candidates instead of crawling the entire catalog deeply.

Default selection:

```text
Top 30 deterministic matches
+ up to 5 uncertain exploration candidates
```

Resolution prefers official employer/ATS pages, then retains useful secondary exact-role pages when official evidence is unavailable. This maximises UI link coverage without mislabelling provenance.

## Related-role discovery

After reranking, SimplyNext searches a small number of top-matching companies for additional official roles. These jobs are marked `source_key=web_discovered`, kept separate from email-originated opportunities, and ranked with the same student profile.

This powers a UI section such as:

```text
You may also like
Reolink — Computer Vision Engineer
Discovered by SimplyNext from company careers
```

## Reliability

- Goh structured tables use deterministic parsing.
- Stage 1 is deterministic.
- Stage 2 uses small batches, individual candidate recovery, rate-limit backoff and explicit missing-assessment fallback.
- Public search uses provider failover; shortlist discovery can aggregate providers for better recall.
- Web failure is non-fatal to ranking.

## Privacy

The dedicated Outlook inbox is an ingestion gateway, not a repository archive. Raw mailbox exports, OAuth tokens, resumes and transcripts must not be committed.

## AWS deployment mapping

The architecture is intentionally deployable without changing the domain model:

```text
Web UI                 -> AWS Amplify
API/workflow endpoints -> API Gateway + Lambda
Resume/transcript      -> S3
Profiles/jobs/results  -> DynamoDB
LLM reasoning          -> Amazon Bedrock
Observability          -> CloudWatch
```

The local Python implementation remains the reference workflow while cloud deployment is added.
