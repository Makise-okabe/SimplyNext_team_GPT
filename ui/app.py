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

from career_agent.student_profile import build_student_profile

DEFAULT_JOBS = PROJECT_ROOT / "data/job_records/latest_matching_candidates.json"
DEFAULT_RESULT = PROJECT_ROOT / "data/matching/career_opportunity_agent.json"
DEFAULT_PROFILE = PROJECT_ROOT / "data/student_profiles/latest_student_profile.json"
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
    for key in ("sn_result", "sn_profile", "sn_logs", "sn_resume_name", "sn_transcript_name"):
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


def _run_backend(resume_path: Path, transcript_path: Path, output_path: Path) -> tuple[dict, list[str]]:
    jobs_path = Path(os.getenv("SIMPLYNEXT_JOBS_PATH", str(DEFAULT_JOBS)))
    if not jobs_path.exists():
        raise FileNotFoundError(
            f"Job catalogue not found at {jobs_path}. Run the email/job ingestion pipeline first."
        )

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
    progress = st.progress(0, text="Preparing your analysis...")
    live_line = st.empty()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        logs.append(line)

        if line.startswith("[1/4]"):
            progress.progress(12, text="Building your student profile...")
        elif line.startswith("[2/4]"):
            progress.progress(32, text="Ranking opportunities from your career inbox...")
        elif "web summary:" in line:
            progress.progress(62, text="Verifying the strongest opportunities on the web...")
        elif line.startswith("[3/4]"):
            progress.progress(74, text="Running semantic fit validation on the top matches...")
        elif line.startswith("[4/4]"):
            progress.progress(91, text="Discovering related roles you may have missed...")

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
        raise RuntimeError("Backend completed without producing the expected result JSON.")
    return _load_json(output_path), logs


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
          <div class="sn-eyebrow">Built for NUS students · Human in the loop</div>
          <h1>Your next opportunity<br><span>is already in your inbox.</span></h1>
          <p>Upload your resume and transcript. SimplyNext turns scattered career emails into a ranked, evidence-backed shortlist tailored to what you can actually do.</p>
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

    action_col, demo_col, _ = st.columns([2.0, 1.6, 4.4])
    with action_col:
        run_clicked = st.button(
            "Find my opportunities →",
            type="primary",
            use_container_width=True,
            disabled=not (resume and transcript),
        )
    with demo_col:
        if st.button(
            "Open latest analysis",
            use_container_width=True,
            disabled=not DEFAULT_RESULT.exists(),
            help="Loads the most recent completed local run without calling the backend again.",
        ):
            st.session_state.sn_result = _load_json(DEFAULT_RESULT)
            if DEFAULT_PROFILE.exists():
                st.session_state.sn_profile = _load_json(DEFAULT_PROFILE)
            st.session_state.sn_logs = []
            st.rerun()

    if run_clicked and resume and transcript:
        try:
            with tempfile.TemporaryDirectory(prefix="simplinext-ui-") as tmp_dir:
                run_dir = Path(tmp_dir)
                resume_path = run_dir / "resume.pdf"
                transcript_path = run_dir / "transcript.pdf"
                output_path = run_dir / "career_opportunity_agent.json"
                resume_path.write_bytes(resume.getbuffer())
                transcript_path.write_bytes(transcript.getbuffer())

                with st.status("Reading your academic and project evidence...", expanded=True) as status:
                    profile = _build_profile(resume_path, transcript_path)
                    status.write(
                        f"Profile ready: {len(profile.get('module_codes') or [])} modules and "
                        f"{len(profile.get('all_skills') or [])} skills detected."
                    )
                    result, logs = _run_backend(resume_path, transcript_path, output_path)
                    status.update(label="Analysis complete", state="complete", expanded=False)

                st.session_state.sn_result = result
                st.session_state.sn_profile = profile
                st.session_state.sn_logs = logs
                st.session_state.sn_resume_name = resume.name
                st.session_state.sn_transcript_name = transcript.name
                st.rerun()
        except Exception as exc:
            st.error("The analysis could not complete.")
            with st.expander("Technical details"):
                st.code(str(exc))

    st.markdown("<div class='sn-flow-title'>What happens behind the screen</div>", unsafe_allow_html=True)
    flow_cols = st.columns(5)
    flow = [
        ("01", "Profile", "Resume + transcript"),
        ("02", "Rank", "All inbox jobs"),
        ("03", "Verify", "Targeted web evidence"),
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
        st.caption("Profile details are unavailable for this saved analysis.")
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


def _backend_job_destination(card: dict) -> tuple[str | None, str | None, bool]:
    """Render exactly what the frozen backend contract provides; invent nothing in the UI."""
    job_page_url = str(card.get("job_page_url") or "").strip()
    if job_page_url:
        kind = str(card.get("job_page_kind") or "").lower()
        confidence = str(card.get("job_page_confidence") or "").lower()
        if kind in {"official_exact", "secondary_exact"} and confidence != "low":
            return "View Job ↗", job_page_url, True
        return "Open Careers Page ↗", job_page_url, True

    fallback_url = str(card.get("search_fallback_url") or "").strip()
    if fallback_url:
        return "Find Job ↗", fallback_url, False

    return None, None, False


def _render_job_cta(card: dict) -> None:
    label, url, resolved = _backend_job_destination(card)
    if url:
        st.link_button(
            label or "Open Job ↗",
            url,
            type="primary" if resolved else "secondary",
            use_container_width=True,
            help=None if resolved else "Search fallback supplied by the frozen backend result.",
        )
        return

    st.markdown(
        "<div class='sn-unavailable'>Job link unavailable</div>",
        unsafe_allow_html=True,
    )


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
        score_col, body_col, action_col = st.columns([1.1, 5.4, 1.8], vertical_alignment="center")
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
    top_matches = list(result.get("top_matches") or [])
    related = list(result.get("related_jobs") or [])

    st.markdown(
        """
        <section class="sn-dashboard-hero">
          <div class="sn-eyebrow">Analysis complete</div>
          <h1>Your strongest opportunities,<br><span>ranked with evidence.</span></h1>
          <p>We searched broadly, enriched selectively, then used semantic review only where it adds value.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Inbox opportunities", int(metrics.get("active_jobs") or 0))
    metric_cols[1].metric("Web enriched", int(metrics.get("web_selected") or 0))
    metric_cols[2].metric("Semantic validated", int(metrics.get("semantic_assessed") or 0))
    metric_cols[3].metric("Related roles found", int(metrics.get("related_jobs_discovered") or 0))

    st.divider()
    _render_profile(profile, result)

    st.divider()
    st.markdown("## Top matches")
    st.caption("Ranked from the full email opportunity set. The top shortlist receives targeted web and semantic enrichment.")
    if not top_matches:
        st.warning("No top matches were produced by this analysis.")
    for rank, card in enumerate(top_matches, start=1):
        _render_match_card(card, rank)

    if related:
        st.divider()
        st.markdown("## You may also like")
        st.caption("Related roles discovered from companies already showing strong fit with your profile.")
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
            st.code("\n".join(logs[-40:]), language="text")


_load_css()
_render_header()

if "sn_result" in st.session_state:
    _render_dashboard(st.session_state.sn_result, st.session_state.get("sn_profile"))
else:
    _render_landing()
