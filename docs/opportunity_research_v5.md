# V5 — Opportunity Provenance & Official-First Research

## Why V5 exists

V1-V4 proved that SimplyNext can ingest a trusted NUS career email, recover a compact job identity, search the public web, avoid false same-job matches, and cache the result. V5 changes the research policy so the system behaves more like a careful human researcher:

1. search the employer's own website / official ATS first;
2. only use LinkedIn, Glassdoor, university career portals and other secondary sources if official evidence is insufficient;
3. distinguish a concrete job posting from a broader programme/opportunity;
4. return the original Outlook email pointer together with research evidence.

## Evidence tiers

### Official

Employer-controlled domains and employer-linked ATS pages.

Examples: `careers.ibm.com`, `careers.marvell.com`, Workday/Greenhouse/Lever pages clearly tied to the employer.

### Institutional

NUS/NTU career portals and other university-hosted sources. These are trustworthy provenance but are not employer-official pages.

### Secondary

LinkedIn, Glassdoor, Indeed, JobStreet and similar public job portals.

### Weak

Unknown pages, transport links and advertising redirects. DuckDuckGo `y.js` and Bing `aclick` ad URLs are explicitly discarded.

## Concrete-job policy

For a concrete job, V5 first gives official candidates an exclusive verification pass. If an exact employer/ATS page is proven, the final status is `verified_exact_job`.

A direct employer careers URL containing a unique `jobId` can be accepted with medium confidence even if a dynamic careers site blocks or times out during fetch, because the trusted source tied that exact official URL to the role.

Only when official evidence fails does V5 search secondary sources. A secondary exact match is labelled `secondary_corroborated`; it is never exposed as an official URL.

## Programme/opportunity policy

A broader internship programme, challenge or recruitment opportunity does not need a fake exact-job page. If the employer's official website confirms the named programme/business unit/context, the result becomes `official_context_supported` and the official page is returned as background evidence.

## Provenance package

The user-facing package contains:

- original source sender and subject;
- Outlook `webLink` pointer back to the original message;
- attachment names;
- official exact-job URL when available;
- official background URLs;
- secondary evidence URLs;
- application URL;
- verification status, basis and confidence.

Raw email bodies, PDF text and fetched-page bodies are not included in the package.

## Canonical live tests

### A. Concrete job — IBM

Source email: `From Your CDE Career Advisors: Industry Opportunities + NUS Career Fest Feb 2026`

Role: `Associate Application Developer-AWS Cloud` (Bangkok)

Official source link contains `careers.ibm.com` and `jobId=88733`.

```powershell
uv run python scripts/run_opportunity_research.py --scan 20 --subject "Industry Opportunities" --company "IBM" --title "Associate Application Developer-AWS Cloud"
```

Expected: `record_kind=job_posting`; official evidence is attempted before any secondary evidence; direct official job URL/jobId is retained.

### B. Concrete job without direct official link — Marvell

Role: `Senior Staff Analog Layout Engineer`, TC ID `280137`.

```powershell
uv run python scripts/run_opportunity_research.py --scan 20 --subject "Industry Opportunities" --company "Marvell" --title "Senior Staff Analog Layout Engineer"
```

Expected: employer-official search first; LinkedIn/Glassdoor/university fallback only if official evidence is insufficient.

### C. Programme/opportunity — McKinsey ILC

```powershell
uv run python scripts/run_opportunity_research.py --scan 10 --subject "McKinsey" --company "McKinsey" --title "Innovation and Learning Centre"
```

Expected: no forced exact-job claim. Employer official ILC context can support the opportunity while the NUS email/PDF remains the original source and Microsoft Forms remains the application endpoint.
