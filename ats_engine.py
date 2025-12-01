import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
from pypdf import PdfReader
import os

# API KEY SETUP: Read key from Streamlit's secrets file
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Fallback for local testing
    if not OPENAI_API_KEY:
        st.error("OpenAI API Key not found. Please set it in secrets or environment variables.")
        st.stop()
        
client = openai.OpenAI(api_key=OPENAI_API_KEY)
# ======================================================
# 1. DOCUMENT PARSING FUNCTION (Unchanged)
# ======================================================

def extract_text_from_pdf(uploaded_file):
    """Reads a Streamlit UploadedFile object and extracts raw text."""
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# ======================================================
# 2. NEW: AI CLEANING & STRUCTURING FUNCTION
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
            temperature=0.0 # Use low temperature for deterministic, cleaning tasks
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during cleaning: {e}"


# ======================================================
# 3. THE RANKING ENGINE FUNCTIONS (Unchanged)
# ======================================================
# NOTE: The ranking functions will now use the clean, structured text.

def get_embedding(text):
    """Converts text into a numeric vector for ranking."""
    text = text.replace("\n", " ") 
    return client.embeddings.create(
        input=[text], 
        model="text-embedding-3-small"
    ).data[0].embedding

def rank_candidates(job_description, candidates_data):
    """Ranks candidates based on semantic similarity to the JD."""
    # ... [Rest of the ranking logic remains the same] ...
    
    jd_vector = get_embedding(job_description)
    scored_candidates = []
    
    for candidate in candidates_data:
        resume_vector = get_embedding(candidate['resume'])
        score = cosine_similarity([jd_vector], [resume_vector])[0][0]
        
        scored_candidates.append({
            "name": candidate['name'],
            "score": score,
            "resume": candidate['resume'] 
        })
    
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    return scored_candidates

# ======================================================
# 4. THE COMPLIANT FEEDBACK ENGINE FUNCTION (Unchanged)
# ======================================================

# Function 4 (The Compliance & Feedback Engine)
def generate_compliant_feedback(job_description, candidate_resume):
    """Generates legally compliant, constructive, and tangible rejection feedback."""
    
    # === NEW SYSTEM PROMPT: THE TANGIBLE FEEDBACK CONSULTANT ===
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