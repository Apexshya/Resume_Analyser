
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Load the trained classifier
clf = pickle.load(open('clf.pkl', 'rb'))
# Load the TF-IDF vectorizer
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Function to clean the resume text
def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Function to predict category from resume
def predict_category(resume_text):
    cleaned_resume = clean_resume(resume_text)
    input_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }
    return category_mapping.get(prediction_id, "Unknown")

# Main function to run the Streamlit app
def main():
    st.title("Resume Category Prediction")

    # File upload section
    uploaded_file = st.file_uploader("Upload a resume (PDF or text file)", type=["pdf", "txt"])

    if uploaded_file is not None:
        file_contents = uploaded_file.getvalue()

        if uploaded_file.type == "text/plain":
            # Text file uploaded
            resume_text = file_contents.decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            # PDF file uploaded
            with pdfplumber.open(uploaded_file) as pdf:
                # Extract text from all pages and concatenate
                resume_text = ""
                for page in pdf.pages:
                    resume_text += page.extract_text()

        # Predict category
        predicted_category = predict_category(resume_text)

        # Display predicted category
        st.subheader("Predicted Category:")
        st.write(predicted_category)

        # Display full resume text
        st.subheader("Full Resume Text:")
        st.write(resume_text)

if __name__ == "__main__":
    main()
