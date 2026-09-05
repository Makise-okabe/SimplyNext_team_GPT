from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from career_agent.student_profile import build_student_profile
from ui.link_rescue import rescue_result_links
from ui.live_pipeline import build_live_matching_candidates

RUNNER = PROJECT_ROOT / "scripts/run_career_opportunity_agent.py"

st.set_page_config(
    page_title="SimplyNext — Career Opportunity Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _load_css() -> None:
    css_path = Path(__file__).with_name("styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _key(company: str | None, title: str | None) -> tuple[str, str]:
    return (
        " ".join(str(company or "").lower().split()),
        " ".join(str(title or "").lower().split()),
    )


def _chips(values: list[str] | tuple[str, ...], *, limit: int | None = None) -> None:
    items = list(values)
    if limit is not None:
        items = items[:limit]
    if not items:
        st.caption("No evidence available")
        return
    markup = "".join(f'<span class="sn-chip">{html.escape(str(item))}</span>' for item in items)
    st.markdown(f'<div class="sn-chip-row">{markup}</div>', unsafe_allow_html=True)


def _bullets(values: list[str] | tuple[str, ...], *, empty: str) -> None:
    if not values:
        st.caption(empty)
        return
    for item in values:
        st.markdown(f"- {item}")


def _reset() -> None:
    for key in (
        "sn_result",
        "sn_profile",
        "sn_logs",
        "sn_resume_name",
        "sn_transcript_name",
        "sn_live_build",
    ):
        st.session_state.pop(key, None)


def _build_profile(resume_path: Path, transcript_path: Path) -> dict:
    profile = build_student_profile(
        resume_text=_pdf_text(resume_path),
        transcript_text=_pdf_text(transcript_path),
        enrich_modules=True,
    )
    return {
        "schema": "simplinext.student_profile.v1",
        "resume_file": resume_path.name,
        "transcript_file": transcript_path.name,
        **profile.to_dict(),
    }


def _run_backend(
    resume_path: Path,
    transcript_path: Path,
    jobs_path: Path,
    output_path: Path,
) -> tuple[dict, list[str]]:
    if not jobs_path.exists():
        raise FileNotFoundError(f"Live matching input was not produced at {jobs_path}.")

    command = [
        sys.executable,
        str(RUNNER),
        "--resume",
        str(resume_path),
        "--transcript",
        str(transcript_path),
        "--jobs",
        str(jobs_path),
        "--semantic-top",
        "5",
        "--top",
        "5",
        "--output",
        str(output_path),
    ]

    logs: list[str] = []
    progress = st.progress(0, text="Ranking live Outlook opportunities...")
    live_line = st.empty()
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_env,
    )

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        logs.append(line)

        if line.startswith("[1/4]"):
            progress.progress(18, text="Building your student profile...")
        elif line.startswith("[2/4]"):
            progress.progress(36, text="Ranking jobs recovered from Outlook...")
        elif "web summary:" in line:
            progress.progress(64, text="Verifying the strongest opportunities on the web...")
        elif line.startswith("[3/4]"):
            progress.progress(76, text="Running semantic fit validation on the top matches...")
        elif line.startswith("[4/4]"):
            progress.progress(92, text="Discovering related roles you may have missed...")

        if line.startswith("[") or "summary:" in line or "Stage 2 batch" in line:
            live_line.caption(line)

    return_code = process.wait()
    live_line.empty()
    if return_code != 0:
        progress.empty()
        tail = "\n".join(logs[-16:])
        raise RuntimeError(f"Career analysis failed.\n\n{tail}")

    progress.progress(100, text="Your opportunity shortlist is ready.")
    if not output_path.exists():
        raise RuntimeError("Frozen backend completed without producing the expected result JSON.")
    return _load_json(output_path), logs


def _attach_source_metadata(result: dict, source_index: dict[tuple[str, str], dict]) -> dict:
    payload = dict(result)
    enriched_cards = []
    source_fields = (
        "primary_source_url",
        "secondary_source_url",
        "official_job_url",
        "application_url",
        "jd_source_url",
        "source_urls",
        "research_status",
        "research_confidence",
        "research_basis",
    )
    for card in payload.get("top_matches") or []:
        enriched = dict(card)
        source = source_index.get(_key(card.get("company"), card.get("title")), {})
        for field in source_fields:
            if field not in enriched or not enriched.get(field):
                enriched[field] = source.get(field)
        enriched_cards.append(enriched)
    payload["top_matches"] = enriched_cards
    return payload


def _render_header() -> None:
    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            "<div class='sn-brand'>Simply<span>Next</span></div>"
            "<div class='sn-brand-sub'>Career Opportunity Agent</div>",
            unsafe_allow_html=True,
        )
    with right:
        if "sn_result" in st.session_state and st.button("New analysis", use_container_width=True):
            _reset()
            st.rerun()


def _render_landing() -> None:
    st.markdown(
        """
        <section class="sn-hero">
          <div class="sn-eyebrow">Built for NUS students · Live Outlook career scan</div>
          <h1>Your next opportunity<br><span>is already in your inbox.</span></h1>
          <p>Upload your resume and transcript. SimplyNext reads the current trusted career emails in Outlook, runs the frozen career pipeline, and returns an evidence-backed shortlist.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Start your career scan")
    left, right = st.columns(2, gap="large")
    with left:
        resume = st.file_uploader(
            "Resume",
            type=["pdf"],
            key="resume_upload",
            help="PDF only. Used to identify explicit skills, projects and experience.",
        )
        st.caption("01 · Skills, projects and experience")
    with right:
        transcript = st.file_uploader(
            "Transcript",
            type=["pdf"],
            key="transcript_upload",
            help="PDF only. NUS modules are enriched into course-derived skills.",
        )
        st.caption("02 · Modules and course-derived skills")

    action_col, _ = st.columns([2.2, 5.8])
    with action_col:
        run_clicked = st.button(
            "Scan Outlook & find matches →",
            type="primary",
            use_container_width=True,
            disabled=not (resume and transcript),
        )

    if run_clicked and resume and transcript:
        try:
            with tempfile.TemporaryDirectory(prefix="simplinext-ui-") as tmp_dir:
                run_dir = Path(tmp_dir)
                resume_path = run_dir / "resume.pdf"
                transcript_path = run_dir / "transcript.pdf"
                live_jobs_path = run_dir / "live_matching_candidates.json"
                output_path = run_dir / "career_opportunity_agent.json"
                resume_path.write_bytes(resume.getbuffer())
                transcript_path.write_bytes(transcript.getbuffer())

                with st.status("Running SimplyNext on your live Outlook inbox...", expanded=True) as status:
                    profile = _build_profile(resume_path, transcript_path)
                    status.write(
                        f"Profile ready: {len(profile.get('module_codes') or [])} modules and "
                        f"{len(profile.get('all_skills') or [])} skills detected."
                    )

                    inbox_lines: list[str] = []

                    def inbox_progress(line: str) -> None:
                        inbox_lines.append(line)
                        status.write(line)

                    live_build = build_live_matching_candidates(
                        live_jobs_path,
                        progress=inbox_progress,
                    )
                    status.write(
                        f"Outlook scan complete: {live_build.email_count} career emails → "
                        f"{live_build.candidate_count} active candidates."
                    )

                    result, backend_logs = _run_backend(
                        resume_path,
                        transcript_path,
                        live_jobs_path,
                        output_path,
                    )
                    result = _attach_source_metadata(result, live_build.source_index)

                    # Link quality is important, but it is presentation enrichment.
                    # A timeout/provider/page error here must never discard an
                    # already-completed ranking result or prevent the dashboard.
                    link_logs: list[str] = []
                    status.write("Validating displayed job links against the live destination pages...")
                    try:
                        result, link_logs = rescue_result_links(result, progress=status.write)
                    except Exception as link_exc:
                        warning = (
                            "Link validation could not complete; showing the ranked results without "
                            f"unverified links. ({type(link_exc).__name__}: {link_exc})"
                        )
                        link_logs.append(warning)
                        status.write(warning)

                    result["live_inbox"] = {
                        "email_count": live_build.email_count,
                        "email_source_counts": live_build.email_source_counts,
                        "raw_job_count": live_build.raw_job_count,
                        "canonical_job_count": live_build.canonical_job_count,
                        "candidate_count": live_build.candidate_count,
                        "candidate_source_counts": live_build.candidate_source_counts,
                    }
                    status.update(label="Live Outlook analysis complete", state="complete", expanded=False)

                st.session_state.sn_result = result
                st.session_state.sn_profile = profile
                st.session_state.sn_logs = [*inbox_lines, *backend_logs, *link_logs]
                st.session_state.sn_live_build = result["live_inbox"]
                st.session_state.sn_resume_name = resume.name
                st.session_state.sn_transcript_name = transcript.name
                st.rerun()
        except Exception as exc:
            st.error("The live Outlook analysis could not complete.")
            with st.expander("Technical details"):
                st.code(str(exc))

    st.markdown("<div class='sn-flow-title'>What happens behind the screen</div>", unsafe_allow_html=True)
    flow_cols = st.columns(5)
    flow = [
        ("01", "Inbox", "All trusted career emails"),
        ("02", "Extract", "Frozen email/JD pipeline"),
        ("03", "Rank", "All active candidates"),
        ("04", "Validate", "Top-5 semantic fit"),
        ("05", "Discover", "Related roles"),
    ]
    for col, (number, title, body) in zip(flow_cols, flow):
        with col:
            st.markdown(
                f"<div class='sn-flow-card'><b>{number}</b><strong>{title}</strong><span>{body}</span></div>",
                unsafe_allow_html=True,
            )


def _render_profile(profile: dict | None, result: dict) -> None:
    st.markdown("## Your profile")
    if not profile:
        st.caption("Profile details are unavailable for this analysis.")
        return

    modules = list(profile.get("module_codes") or [])
    skills = list(profile.get("all_skills") or [])
    explicit = list(profile.get("explicit_skills") or [])
    course_skills = list(profile.get("course_derived_skills") or [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Courses detected", len(modules))
    c2.metric("Total skills", len(skills))
    c3.metric("Resume skills", len(explicit))
    c4.metric("Course-derived", len(course_skills))

    with st.expander("Explore detected evidence"):
        st.markdown("**Skills**")
        _chips(skills)
        st.markdown("**NUS modules**")
        _chips(modules)

    resume_name = result.get("resume_file") or profile.get("resume_file")
    transcript_name = result.get("transcript_file") or profile.get("transcript_file")
    if resume_name or transcript_name:
        st.caption(f"Evidence sources: {resume_name or 'resume'} · {transcript_name or 'transcript'}")


def _unique_links(card: dict) -> list[tuple[str, str]]:
    job_page_url = str(card.get("job_page_url") or "").strip()
    kind = str(card.get("job_page_kind") or "").lower()

    ordered: list[tuple[str, str | None]] = []
    if job_page_url and kind.startswith("official"):
        ordered.append(("Primary job page ↗", job_page_url))
    elif job_page_url and kind.startswith("secondary"):
        ordered.append(("Secondary job page ↗", job_page_url))
    elif job_page_url and kind == "company_careers":
        ordered.append(("Company careers ↗", job_page_url))
    elif job_page_url and kind != "unresolved":
        ordered.append(("View job ↗", job_page_url))

    ordered.extend(
        [
            ("Apply ↗", card.get("application_url")),
            ("Official job page ↗", card.get("official_job_url")),
            ("Primary source ↗", card.get("primary_source_url")),
            ("Secondary source ↗", card.get("secondary_source_url")),
        ]
    )

    seen: set[str] = set()
    links: list[tuple[str, str]] = []
    for label, raw_url in ordered:
        url = str(raw_url or "").strip()
        if not url or url in seen:
            continue
        lowered = url.lower()
        if "google.com/search" in lowered or "bing.com/search" in lowered or "duckduckgo.com/" in lowered:
            continue
        seen.add(url)
        links.append((label, url))
    return links


def _render_job_cta(card: dict) -> None:
    links = _unique_links(card)
    if links:
        for index, (label, url) in enumerate(links[:2]):
            st.link_button(
                label,
                url,
                type="primary" if index == 0 else "secondary",
                use_container_width=True,
            )
        return

    st.markdown("<div class='sn-unavailable'>No verified direct job page found</div>", unsafe_allow_html=True)


def _evidence_label(card: dict) -> str:
    level = str(card.get("evidence_level") or "source_only").lower()
    labels = {
        "full_jd": "Full JD evidence",
        "partial_jd": "Partial JD evidence",
        "source_only": "Email evidence",
    }
    return labels.get(level, level.replace("_", " ").title())


def _render_match_card(card: dict, rank: int) -> None:
    score = float(card.get("final_score") or card.get("score") or 0)
    company = str(card.get("company") or "Unknown company")
    title = str(card.get("title") or "Untitled role")
    fit = str(card.get("fit_label") or "possible").title()
    confidence = str(card.get("confidence") or "low").title()
    evidence = _evidence_label(card)

    with st.container(border=True):
        score_col, body_col, action_col = st.columns([1.1, 5.1, 2.1], vertical_alignment="center")
        with score_col:
            st.markdown(
                f"<div class='sn-score'><span>{score:.0f}</span><small>% match</small></div>",
                unsafe_allow_html=True,
            )
        with body_col:
            st.caption(f"#{rank} · {company}")
            st.markdown(f"### {title}")
            st.markdown(
                f"<div class='sn-meta'><span>{html.escape(fit)} fit</span>"
                f"<span>{html.escape(confidence)} confidence</span>"
                f"<span>{html.escape(evidence)}</span></div>",
                unsafe_allow_html=True,
            )
        with action_col:
            _render_job_cta(card)

        st.write(str(card.get("why_match") or "No semantic explanation available."))

        matched = list(card.get("matched_resume_skills") or []) + list(card.get("matched_course_skills") or [])
        if matched:
            st.markdown("**Matched skills**")
            _chips(list(dict.fromkeys(matched)), limit=10)

        with st.expander("Why this ranking"):
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("**Supporting evidence**")
                _bullets(list(card.get("matched_evidence") or []), empty="No semantic evidence returned.")
            with right:
                st.markdown("**Missing / weaker evidence**")
                _bullets(
                    list(card.get("missing_or_weak_evidence") or []),
                    empty="No material gap was identified.",
                )

            inferred = list(card.get("inferred_job_skills") or [])
            if inferred:
                st.markdown("**Role skills considered**")
                _chips(inferred, limit=12)
            st.caption(
                "SimplyNext separates resume evidence, course-derived evidence and title-inferred role skills. "
                "Sparse job descriptions lower confidence rather than inventing requirements."
            )


def _render_related_card(card: dict) -> None:
    company = str(card.get("company") or "Unknown company")
    title = str(card.get("title") or "Untitled role")
    score = float(card.get("score") or card.get("final_score") or 0)

    with st.container(border=True):
        st.caption(company)
        st.markdown(f"#### {title}")
        st.markdown(f"**{score:.0f}% match**")
        st.caption(card.get("recommendation_reason") or "Related role from a strongly matching company.")
        _render_job_cta(card)


def _render_dashboard(result: dict, profile: dict | None) -> None:
    metrics = result.get("metrics") or {}
    live = result.get("live_inbox") or {}
    top_matches = list(result.get("top_matches") or [])
    related = list(result.get("related_jobs") or [])

    st.markdown(
        """
        <section class="sn-dashboard-hero">
          <div class="sn-eyebrow">Live Outlook analysis complete</div>
          <h1>Your strongest opportunities,<br><span>ranked with evidence.</span></h1>
          <p>The shortlist below was built from the career emails currently recovered from your connected Outlook inbox.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Career emails", int(live.get("email_count") or 0))
    metric_cols[1].metric("Active candidates", int(live.get("candidate_count") or metrics.get("active_jobs") or 0))
    metric_cols[2].metric("Web enriched", int(metrics.get("web_selected") or 0))
    metric_cols[3].metric("Semantic validated", int(metrics.get("semantic_assessed") or 0))
    metric_cols[4].metric("Related roles", int(metrics.get("related_jobs_discovered") or 0))

    source_counts = live.get("email_source_counts") or {}
    if source_counts:
        st.caption(
            f"Live inbox sources: {int(source_counts.get('goh_ze_li') or 0)} Goh Ze Li email(s) · "
            f"{int(source_counts.get('talentconnect') or 0)} TalentConnect email(s)"
        )

    st.divider()
    _render_profile(profile, result)

    st.divider()
    st.markdown("## Top matches")
    st.caption("Ranked by the frozen career-opportunity backend from this run's live Outlook job catalogue.")
    if not top_matches:
        st.warning("No top matches were produced by this analysis.")
    for rank, card in enumerate(top_matches, start=1):
        _render_match_card(card, rank)

    if related:
        st.divider()
        st.markdown("## You may also like")
        st.caption("Related roles discovered by the frozen backend from companies already showing strong fit.")
        columns = st.columns(min(3, len(related)), gap="large")
        for index, card in enumerate(related[:6]):
            with columns[index % len(columns)]:
                _render_related_card(card)

    st.markdown(
        """
        <div class="sn-footer-note">
          SimplyNext recommends; you decide. No automatic applications, no fabricated requirements, and no hidden career decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    logs = st.session_state.get("sn_logs") or []
    if logs:
        with st.expander("Demo diagnostics"):
            st.code("\n".join(logs[-120:]), language="text")


_load_css()
_render_header()

if "sn_result" in st.session_state:
    _render_dashboard(st.session_state.sn_result, st.session_state.get("sn_profile"))
else:
    _render_landing()
