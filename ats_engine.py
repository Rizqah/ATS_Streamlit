import os
import io

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from dotenv import load_dotenv

# Try to import Streamlit if available
try:
    import streamlit as st
except ImportError:
    st = None

# Load .env for local development
load_dotenv()

# -------------------------
# OpenAI client setup
# -------------------------
def get_openai_api_key() -> str:
    """
    Get the OpenAI API key from:
    1. Streamlit secrets (Cloud / local Streamlit)
    2. Environment variables / .env (local, CI, etc.)
    """

    # 1. Try Streamlit secrets if Streamlit is available
    if st is not None:
        try:
            key = st.secrets["OPENAI_API_KEY"]
            if key:
                return key
        except Exception:
            # st.secrets may not be set or key missing – fall back to env
            pass

    # 2. Try environment variables (.env / system)
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # 3. Nothing configured: show friendly error in Streamlit or raise
    msg = (
        "OpenAI API Key not found.\n"
        "Please set OPENAI_API_KEY in Streamlit secrets (on Cloud) "
        "or in a local .env / environment variable."
    )

    if st is not None:
        st.error(msg)
        st.stop()

    raise RuntimeError(msg)


OPENAI_API_KEY = get_openai_api_key()

# ===== OpenAI client (new SDK style) =====
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


# ======================================================
# 1. DOCUMENT PARSING FUNCTION
# ======================================================
def extract_text_from_pdf(uploaded_file):
    """Reads a Streamlit UploadedFile object and extracts raw text."""
    # Make sure we're at the beginning of the file
    uploaded_file.seek(0)
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


# ======================================================
# 2. AI CLEANING & STRUCTURING FUNCTION
# ======================================================
def clean_and_structure_resume(raw_resume_text):
    """Uses LLM to clean noise and apply section tags to text."""
    
    system_prompt = """
    You are an expert Document Processor. Your task is to clean up raw, noisy text extracted from a resume.

    INSTRUCTIONS:
    1. Remove all noise: page numbers, headers, footers, repetitive lines, and obvious contact information (phone numbers, email addresses, URLs).
    2. Structure the remaining core content using the following tags only: [SUMMARY], [SKILLS], [EXPERIENCE], [EDUCATION].
    3. Return only the cleaned and tagged text. DO NOT add any extra commentary or introductory phrases.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_resume_text}
            ],
            temperature=0.0  # deterministic cleaning
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during cleaning: {e}"


# ======================================================
# 3. RANKING ENGINE FUNCTIONS
# ======================================================
def get_embedding(text):
    """Converts text into a numeric vector for ranking."""
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    ).data[0].embedding
    return embedding


def rank_candidates(job_description, candidates_data):
    """
    Ranks candidates based on semantic similarity to the JD.
    candidates_data: list[{"name": str, "resume": str}]
    """
    jd_vector = get_embedding(job_description)
    scored_candidates = []
    
    for candidate in candidates_data:
        resume_vector = get_embedding(candidate["resume"])
        score = cosine_similarity([jd_vector], [resume_vector])[0][0]
        
        scored_candidates.append({
            "name": candidate["name"],
            "score": score,
            "resume": candidate["resume"]
        })
    
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates


# ======================================================
# 4. COMPLIANT FEEDBACK ENGINE
# ======================================================
def generate_compliant_feedback(job_description, candidate_resume):
    """Generates legally compliant, constructive, and tangible rejection feedback."""
    
    system_prompt = """
    You are an Expert Resume Consultant and a Compliance Officer. Your primary goal is to provide **highly specific, tangible, and constructive feedback** based *only* on the content of the resume and the requirements of the job description (JD).

    INSTRUCTIONS FOR TANGIBLE FEEDBACK:
    1.  **Analyze the Weak Link:** Identify the single biggest gap where the candidate mentioned a required hard skill but failed to demonstrate sufficient depth, context, or quantifiable results required by the JD.
    2.  **Focus on Specificity:** Instead of saying "lacks Python," say, "lacks demonstrated experience using Python for **data pipeline automation** as the JD requires."
    3.  **Provide Actionable Advice:** Offer one concrete, actionable suggestion for how they can re-write or strengthen the *existing* experience on their resume to better match the JD's focus (e.g., "Add metrics showing efficiency gains").

    THE "RED ZONE" (STRICTLY FORBIDDEN—Legal Compliance):
    - Do NOT mention: Personality, tone, enthusiasm, "culture fit," age, gender, or soft skills.
    
    THE "GREEN ZONE" (ONLY USE THESE):
    - Hard Skills, Objective Metrics, Demonstrated Specificity, and Mismatched Depth.

    Write the polite and legally safe rejection email using this structured, tangible advice.
    """

    user_prompt = f"""
    JOB DESCRIPTION:
    {job_description}

    CLEANED CANDIDATE RESUME:
    {candidate_resume}

    Write the rejection email.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"



# ... (Existing imports and functions) ...

def extract_text_from_docx(uploaded_file):
    """Reads a Streamlit UploadedFile object and extracts text from DOCX."""
    
    # docx library needs a file path or file-like object; we use the file stream
    document = docx.Document(uploaded_file)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + '\n'
        
    return text.strip()

