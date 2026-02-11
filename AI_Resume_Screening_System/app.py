import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening", page_icon="📄")

st.title("📄 AI-Based Resume Screening System")
st.write("Compare multiple resumes with a job description using AI")

# Upload multiple resumes
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Job Description
job_desc = st.text_area("Paste Job Description")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

if st.button("🔍 Analyze Resumes"):
    if uploaded_files and job_desc:

        scores = []

        for file in uploaded_files:
            resume_text = extract_text_from_pdf(file)

            documents = [resume_text, job_desc]

            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(documents)

            similarity = cosine_similarity(
                tfidf_matrix[0:1], tfidf_matrix[1:2]
            )

            match_score = similarity[0][0] * 100

            scores.append({
                "Resume": file.name,
                "Match Score (%)": round(match_score, 2)
            })

        # Convert to DataFrame
        df = pd.DataFrame(scores)

        # Sort by score
        df = df.sort_values(
            by="Match Score (%)", 
            ascending=False
        )

        st.success("✅ Resume Analysis Complete!")

        st.dataframe(df)

        # Highlight best resume
        best_resume = df.iloc[0]

        st.markdown("## 🏆 Best Matching Resume")
        st.write(
            f"**{best_resume['Resume']}** "
            f"with **{best_resume['Match Score (%)']}%** match"
        )

    else:
        st.error("Please upload resumes and paste job description.")
