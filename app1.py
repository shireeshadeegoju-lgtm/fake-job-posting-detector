import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack
import os
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="wide"
)

st.write("Gemini API Loaded:", True)

def generate_ai_summary(text):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(
            f"Summarize this job posting in 4-5 simple lines:\n\n{text}"
        )
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"
def generate_red_flag_explanations(text):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(
            f"""
            You are an expert in detecting fraudulent job postings.
            Analyze the following job post and list 5–7 clear red flags if it appears fake.
            Each red flag must be short and specific, with a one-line explanation.

            Job Posting:
            {text}

            Return the output in bullet points.
            """
        )
        return response.text
    except Exception as e:
        return f"Error generating red flags: {e}"




# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    try:
        model_dir = 'dupmodel/model'  # Match your notebook's folder name
        
        with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
        return model, vectorizer, model_info
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Error: {e}")
        st.info("Please run the Jupyter notebook first to train and save the model.")
        return None, None, None

model, vectorizer, model_info = load_model()

# Check if model is loaded
if model is None:
    st.stop()

# Header
st.title(" Fake Job Posting Detector")
st.markdown("Enter job posting details to check if it's legitimate or fake.")

# Input form
st.subheader(" Enter Job Posting Details")

with st.form("job_posting_form"):
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
        location = st.text_input("Location", placeholder="e.g., New York, NY")
        department = st.text_input("Department", placeholder="e.g., Engineering")
    
    with col2:
        employment_type = st.selectbox(
            "Employment Type",
            ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
        )
        required_experience = st.selectbox(
            "Required Experience",
            ["", "Entry level", "Mid-Senior level", "Director", "Executive", "Internship"]
        )
        required_education = st.selectbox(
            "Required Education",
            ["", "High School", "Bachelor's Degree", "Master's Degree", "Doctorate", "Unspecified"]
        )
    
    # Text areas
    company_profile = st.text_area(
        "Company Profile",
        placeholder="Brief company description...",
        height=100
    )
    
    description = st.text_area(
        "Job Description *",
        placeholder="Enter the full job description...",
        height=150
    )
    
    col3, col4 = st.columns(2)
    with col3:
        requirements = st.text_area("Requirements", height=100)
    with col4:
        benefits = st.text_area("Benefits", height=100)
    
    # Additional fields
    col5, col6 = st.columns(2)
    with col5:
        industry = st.text_input("Industry", placeholder="e.g., Technology")
        function = st.text_input("Function", placeholder="e.g., Engineering")
    with col6:
        salary_range = st.text_input("Salary Range", placeholder="e.g., $60k-$80k")
    
    # Checkboxes
    col7, col8, col9 = st.columns(3)
    with col7:
        has_company_logo = st.checkbox("Has Company Logo")
    with col8:
        has_questions = st.checkbox("Has Screening Questions")
    with col9:
        telecommuting = st.checkbox("Remote/Telecommuting")
    
    # Submit button
    submitted = st.form_submit_button("🔍 Analyze Job Posting", use_container_width=True)

# Process prediction
if submitted:
    if not description.strip():
        st.error("⚠️ Please enter a job description!")
    else:
        with st.spinner("🔄 Analyzing..."):
            # Combine all text fields (same as training)
            combined_text = f"{title} {location} {department} {company_profile} {description} {requirements} {benefits} {employment_type} {required_experience} {required_education} {industry} {function}"
            
            # Calculate numeric features (same as training)
            text_length = len(combined_text)
            word_count = len(combined_text.split())
            has_salary = 1 if salary_range.strip() else 0
            
            # Create numeric features array
            numeric_features = np.array([[
                text_length,
                word_count,
                int(has_company_logo),
                int(has_questions),
                int(telecommuting),
                has_salary
            ]])
            
            # Transform text using TF-IDF
            text_tfidf = vectorizer.transform([combined_text])
            
            # Combine features (IMPORTANT: same order as training!)
            combined_features = hstack([text_tfidf, numeric_features])
            
            # Make prediction
            prediction = model.predict(combined_features)[0]
            prediction_proba = model.predict_proba(combined_features)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Main result
        if prediction == 1:
            st.error("### FAKE POSTING DETECTED!")
            st.subheader("🔎 AI Red Flag Explanations")
            with st.spinner("Analyzing red flags using AI..."):
                red_flags = generate_red_flag_explanations(combined_text)
            st.warning(red_flags)
        else:
            st.success("###  LEGITIMATE POSTING")

        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Prediction", "FAKE" if prediction == 1 else "REAL")
        with col_m2:
            st.metric("Confidence", f"{max(prediction_proba)*100:.1f}%")
        with col_m3:
            st.metric("Text Length", f"{text_length} chars")
        
        # Probability breakdown
        st.markdown("### 📈 Probability Breakdown")
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.info("**Legitimate Probability**")
            st.progress(float(prediction_proba[0]))
            st.write(f"**{prediction_proba[0]*100:.2f}%**")
        st.markdown("### 🧠 AI Summary of Job Posting")
        with st.spinner("Generating AI summary..."):
            ai_summary = generate_ai_summary(combined_text)
        st.info(ai_summary)

        with col_p2:
            st.warning("**Fraudulent Probability**")
            st.progress(float(prediction_proba[1]))
            st.write(f"**{prediction_proba[1]*100:.2f}%**")
        
        # Recommendations
        st.markdown("---")
        if prediction == 1:
            st.error("""
            ### 🚨 Warning: Potential Red Flags
            
            This job posting shows characteristics of fraudulent listings:
            - ⚠️ Vague or suspicious language patterns
            - ⚠️ Unusual content structure
            - ⚠️ Missing critical information
            
            **What to do:**
            1. Research the company thoroughly
            2. Verify on official company website
            3. Never pay money for job applications
            4. Be cautious with personal information
            """)
        else:
            st.success("""
            ### ✅ This Posting Appears Legitimate
            
            However, always practice due diligence:
            - Verify the company exists
            - Check reviews on Glassdoor
            - Visit the official company website
            - Trust your instincts
            """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p><strong>Model Used:</strong> {model_info['model_name'] if model_info else 'Unknown'}</p>
    <p>⚠️ This tool provides predictions based on patterns. Always verify independently.</p>
</div>
""", unsafe_allow_html=True)
