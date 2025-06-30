import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()  

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities


# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")
st.write("This application allows you to upload PDF resumes and rank them based on a job description.")

# Job description input
st.header("Job Description")
job_description= st.text_area ("Enter the job description")


# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)


if uploaded_files and job_description:
    st.header("Ranking Resumes")
    st.write("The resumes will be ranked based on their relevance to the job description.")

    st.header("Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores })
    results = results.sort_values(by="Score", ascending=False)

    # Download button for CSV
    csv = results.to_csv(index=False)
    st.download_button("Download Results as CSV", csv, "ranked_resumes.csv", "text/csv")
    
    st.bar_chart(results.set_index("Resume")["Score"])

    

    
    st.bar_chart(results.set_index("Resume")["Score"])
    st.write(results)
