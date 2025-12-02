import streamlit as st
import pandas as pd
import os
import openai
from ats_engine import (
    rank_candidates, 
    generate_compliant_feedback, 
    extract_text_from_pdf,
    extract_text_from_docx,
    clean_and_structure_resume 
) 

# --- API Key Setup (Ensure st.secrets is imported or defined in ats_engine) ---

# Re-initialize the client globally here to ensure Streamlit can find the secret
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    # Fallback for local testing (Ensure you set an environment variable)
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
    if not OPENAI_API_KEY:
        st.error("OpenAI API Key not found. Please set it in st.secrets or environment variables.")
        st.stop()
        
client = openai.OpenAI(api_key=OPENAI_API_KEY)


# --- 1. Main UI and Tabs Setup ---

st.set_page_config(page_title="Compliant ATS Dashboard", layout="wide")
st.title("🤖 Recruiter ATS Dashboard")
st.sidebar.markdown("# Recruiter Dashboard") # Placeholder for multi-page structure

# Define the tabs
tab1, tab2, tab3 = st.tabs(["⚙️ Setup & Upload", "📊 Ranking & Scores", "📧 Feedback Generator"])

# Initialize session state for storing results across tabs
if 'ranked_data' not in st.session_state:
    st.session_state['ranked_data'] = None

# --- TAB 1: Setup & Upload ---
with tab1:
    st.header("1. Define Job & Gather Resumes")

    # Use columns for a cleaner side-by-side input layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the Full Job Description Here:", 
            height=300,
            key="job_desc_input",
            value="We need a Chief Financial Officer (CFO). Must have CPA certification. Experience managing large corporate budgets. Strategic financial planning."
        )

    with col2:
        st.subheader("Candidate Resumes")
        # --- CHANGE 1: Added 'doc' to accepted types ---
        uploaded_files = st.file_uploader(
            "Upload Resumes (PDF, DOCX, and DOC supported):", 
            type=['pdf', 'docx', 'doc'], 
            accept_multiple_files=True
        )

    st.markdown("---")
    
    # Ranking Trigger Button
    if uploaded_files and st.button("🚀 Run Ranking Engine", type="primary"):
        if not job_description:
            st.error("Please paste the Job Description before running the engine.")
            st.stop()
            
        with st.spinner("Processing files, cleaning with AI, and running Vector Embedding analysis..."):
            
            candidate_list_for_ranking = []
            
            for file in uploaded_files:
                file_extension = file.name.split('.')[-1].lower()
                raw_resume_text = ""

                # Step 1: Extract RAW text (Multi-format handling)
                if file_extension == 'pdf':
                    raw_resume_text = extract_text_from_pdf(file)
                elif file_extension == 'docx':
                    file.seek(0)
                    raw_resume_text = extract_text_from_docx(file)
                # --- CHANGE 2: Handling for older .doc files ---
                elif file_extension == 'doc':
                    st.error(f"⚠️ **Skipping {file.name}:** The legacy '.doc' format is unsupported on this cloud environment. Please convert this file to a modern '.docx' or '.pdf' and re-upload.")
                    continue
                else:
                    st.warning(f"Skipping unsupported file type: {file.name}")
                    continue 
                
                if raw_resume_text:
                    # Step 2: Clean and Structure the text with AI
                    clean_resume_text = clean_and_structure_resume(raw_resume_text)
                    
                    candidate_list_for_ranking.append({
                        "name": file.name,
                        "resume": clean_resume_text
                    })

            st.session_state['job_description'] = job_description
            
            if candidate_list_for_ranking:
                st.info(f"Successfully processed and cleaned {len(candidate_list_for_ranking)} resumes.")
                
                # Step 3: Call the ranking function
                ranking_results = rank_candidates(job_description, candidate_list_for_ranking)
                st.session_state['ranked_data'] = ranking_results
                st.success("Ranking Complete! See the **Ranking & Scores** tab.")
            else:
                st.warning("No valid files were processed.")
        
# --- TAB 2: Ranking & Scores ---
with tab2:
    st.header("2. Candidate Ranking Results")

    if st.session_state['ranked_data'] is not None:
        ranking_results = st.session_state['ranked_data']
        
        # Display results in a clean table
        df = pd.DataFrame(ranking_results)
        df['Score'] = (df['score'] * 100).round(1).astype(str) + '%'
        
        # Select relevant columns for display
        df_display = df[['name', 'Score']].rename(columns={'name': 'Candidate'})
        
        st.subheader("Semantic Match Scoreboard")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.info("The table is sorted by score (highest match first).")

        st.subheader("Review Cleaned Resume Text")
        # Allow recruiter to quickly check the clean text
        candidate_names = [r['name'] for r in ranking_results]
        selected_name = st.selectbox("Select Candidate to Review:", candidate_names)
        
        selected_candidate = next((r for r in ranking_results if r['name'] == selected_name), None)
        
        if selected_candidate:
            with st.expander(f"Cleaned Resume Text for {selected_name}"):
                st.code(selected_candidate['resume'], language='markdown')

    else:
        st.warning("Please run the ranking engine in the 'Setup & Upload' tab first.")

# --- TAB 3: Feedback Generator ---
with tab3:
    st.header("3. Generate Compliant Rejection Feedback")

    if st.session_state['ranked_data'] is not None:
        ranking_results = st.session_state['ranked_data']
        job_description = st.session_state['job_description']
        
        # Target the lowest scoring candidate for rejection (last element in sorted list)
        candidate_to_reject = ranking_results[-1]
        
        st.info(f"Targeting **{candidate_to_reject['name']}** (Lowest Score: {(candidate_to_reject['score'] * 100):.1f}%) for Compliant Feedback.")
        
        if st.button(f"✍️ Draft Email for {candidate_to_reject['name']}"):
            
            with st.spinner("Generating Tangible, Legally Compliant Draft..."):
                
                feedback_draft = generate_compliant_feedback(
                    job_description, 
                    candidate_to_reject['resume']
                )
            
            st.subheader("Final Draft (Recruiter Review Required)")
            st.code(feedback_draft, language='text')

            if st.checkbox("Recruiter Review: I confirm this feedback is safe and accurate."):
                st.success("✅ Email ready to send! Liability risk minimized.")
                st.download_button(
                    label="Download Draft",
                    data=feedback_draft,
                    file_name=f"Rejection_Email_{candidate_to_reject['name'].replace('.', '_')}.txt",
                    mime="text/plain"
                )
    else:
        st.warning("Please run the ranking engine in the 'Setup & Upload' tab first.")
