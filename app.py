import streamlit as st
import pandas as pd

from ats_engine import (
    rank_candidates,
    generate_compliant_feedback,
    extract_text_from_pdf,
    clean_and_structure_resume,
    compute_fit_score,
    rewrite_resume,
)

st.set_page_config(page_title="Compliant ATS Matcher", layout="wide")

# --- ROLE SELECTION ---
with st.sidebar:
    st.title("ATS Prototype")
    role = st.radio("I am a:", ["Recruiter", "Applicant"])

st.title("🤖 Compliant ATS Prototype")

# =========================================
# RECRUITER MODE
# =========================================
if role == "Recruiter":
    st.subheader("Recruiter Dashboard – Rank Candidates & Generate Compliant Feedback")

    # --- 1. INPUT AREA ---
    st.header("1. Define the Job & Upload Resumes")
    job_description = st.text_area(
        "Paste the Full Job Description Here:",
        height=200,
        value="We need a Chief Financial Officer (CFO). Must have CPA certification. Experience managing large corporate budgets. Strategic financial planning.",
    )

    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF Only for this MVP):",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # --- 2. RANKING ENGINE TRIGGER ---
    if uploaded_files and st.button("🚀 Run Ranking Engine", type="primary"):
        with st.spinner("Processing files, cleaning with AI, and running analysis..."):
            candidate_list_for_ranking = []

            for file in uploaded_files:
                # Step 1: Extract RAW text
                raw_resume_text = extract_text_from_pdf(file)

                # Step 2: Clean and Structure the text with AI
                clean_resume_text = clean_and_structure_resume(raw_resume_text)

                candidate_list_for_ranking.append(
                    {
                        "name": file.name,
                        "resume": clean_resume_text,
                    }
                )

            st.info(
                f"Successfully processed and cleaned {len(candidate_list_for_ranking)} resumes."
            )

            ranking_results = rank_candidates(job_description, candidate_list_for_ranking)

            df = pd.DataFrame(ranking_results)
            df["Score"] = (df["score"] * 100).round(1).astype(str) + "%"
            df = df[["name", "Score"]].rename(columns={"name": "Candidate"})

            st.success("Ranking Complete!")
            st.dataframe(df, use_container_width=True)

            st.session_state["ranked_data"] = ranking_results

    # --- 3. FEEDBACK ENGINE TRIGGER ---
    if "ranked_data" in st.session_state:
        st.header("3. Generate Compliant Feedback")
        candidate_to_reject = st.session_state["ranked_data"][-1]
        st.info(
            f"Targeting **{candidate_to_reject['name']}** (Lowest Score) for Rejection Feedback."
        )
        if st.button(f"✍️ Draft Email for {candidate_to_reject['name']}"):
            st.warning("Generating Legally Compliant Draft...")
            feedback_draft = generate_compliant_feedback(
                job_description, candidate_to_reject["resume"]
            )
            st.subheader("Final Draft (Human-in-the-Loop Review)")
            st.code(feedback_draft, language="text")
            if st.checkbox(
                "Recruiter Review: I confirm this feedback is safe and accurate."
            ):
                st.success("✅ Email ready to send! Liability risk minimized.")


# =========================================
# APPLICANT MODE
# =========================================
else:
    st.subheader("Applicant Dashboard – Check Fit & Improve Your Resume")

    st.markdown(
        "Upload your resume and paste the job description to see your ATS fit score "
        "and get an AI-optimised version of your resume."
    )

    col1, col2 = st.columns(2)

    with col1:
        jd_applicant = st.text_area(
            "Paste the Job Description:",
            height=260,
            placeholder="Paste the job description you are applying for...",
        )

    with col2:
        resume_file = st.file_uploader(
            "Upload Your Resume (PDF preferred):",
            type=["pdf"],
        )
        manual_resume_text = st.text_area(
            "Or paste your resume text here:",
            height=260,
            placeholder="If you don't have a PDF, paste your resume content here...",
        )

    analyze_button = st.button("🔍 Analyse & Improve My Resume", type="primary")

    if analyze_button:
        if not jd_applicant:
            st.error("Please paste the Job Description first.")
        else:
            # Get resume text either from PDF or text area
            if resume_file is not None:
                raw_resume = extract_text_from_pdf(resume_file)
            elif manual_resume_text.strip():
                raw_resume = manual_resume_text.strip()
            else:
                st.error("Please upload a resume PDF or paste your resume text.")
                st.stop()

            with st.spinner("Analysing your resume and generating improvements..."):
                # Optional: clean & structure text like in recruiter mode
                cleaned_resume = clean_and_structure_resume(raw_resume)

                # 1. Compute fit score
                score = compute_fit_score(jd_applicant, cleaned_resume)

                # 2. Rewrite resume
                optimised_resume_md = rewrite_resume(jd_applicant, cleaned_resume)

            # --- OUTPUT SECTION ---
            st.success("Analysis complete! Scroll down to see your results.")

            # Show fit score
            st.header("1. ATS Fit Score")
            score_percent = max(0.0, min(1.0, score)) * 100  # clip to 0–100
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.metric("Overall Match", f"{score_percent:.1f}%")
            with col_b:
                st.progress(score_percent / 100.0)

            st.caption(
                "This score is based on how semantically similar your resume is to the job description "
                "using AI embeddings. Higher is better, but even a lower score can often be improved."
            )

            # Show optimised resume
            st.header("2. Suggested Optimised Resume")
            st.markdown(
                "Below is an improved version of your resume based on the job description. "
                "Review carefully before using – make sure everything is accurate and honest."
            )
            st.markdown(optimised_resume_md)

            # Optional download button (simple text version)
            st.download_button(
                label="📩 Download Optimised Resume (Markdown)",
                data=optimised_resume_md,
                file_name="optimised_resume.md",
                mime="text/markdown",
            )
