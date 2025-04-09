import os
os.environ["TF_KERAS"] = "1"  # if you need tf-keras, but if using torch this may not be necessary
import torch
print("Torch version:", torch.__version__)
import re
import random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tf_keras as keras
from collections import Counter
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ------------------------------
# Load Synthetic EHR Data
# ------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("data/synthetic_ehr_data.csv", parse_dates=["visit_date"])
    return df

df = load_data()

# ------------------------------
# Define an Extensive Stopword List for Text Analysis
# ------------------------------
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
extra_stops = {
    "patient", "doctor", "hospital", "clinic", "mg", "dr", "mr", "mrs", "ms", 
    "with", "without", "the", "and", "to", "a", "of", "is", "was", "in", "for", 
    "it", "my", "at", "fine", "better", "comes", "out", "no", "again", "all", "upon", 
    "presents", "questions", "answers", "past", "etc", "each", "also", "very", "visit", 
    "visits", "exam", "follow", "up", "check", "checkup", "year", "old", "history", "plan", 
    "months", "male", "female"
}
# IMPORTANT: TfidfVectorizer requires stop_words as a list or string; so we cast to list:
stopwords = list(set(ENGLISH_STOP_WORDS).union(extra_stops))


st.title("📊 Synthetic EHR Dashboard with Enhanced NLP & Clustering Analysis")

# ------------------------------
# Sidebar Filters and Reset Button
# ------------------------------
st.sidebar.header("Data Filters")
all_departments = sorted(df["department"].dropna().unique().tolist())
if "dept_filter" not in st.session_state:
    st.session_state["dept_filter"] = all_departments
dept_selection = st.sidebar.multiselect("Department:", options=all_departments, default=st.session_state["dept_filter"], key="dept_filter")

all_genders = sorted(df["gender"].dropna().unique().tolist())
if "gender_filter" not in st.session_state:
    st.session_state["gender_filter"] = all_genders
gender_selection = st.sidebar.multiselect("Gender:", options=all_genders, default=st.session_state["gender_filter"], key="gender_filter")

all_physicians = sorted(df["physician_id"].dropna().unique().tolist())
if "phys_filter" not in st.session_state:
    st.session_state["phys_filter"] = all_physicians
phys_selection = st.sidebar.multiselect("Physician ID:", options=all_physicians, default=st.session_state["phys_filter"], key="phys_filter")

if "age_filter" not in st.session_state:
    st.session_state["age_filter"] = sorted(df["age_group"].dropna().unique().tolist())
age_group_selection = st.sidebar.multiselect("Age Group:", options=st.session_state["age_filter"], default=st.session_state["age_filter"], key="age_filter")

min_date = df["visit_date"].min().date()
max_date = df["visit_date"].max().date()
if "date_filter" not in st.session_state:
    st.session_state["date_filter"] = (min_date, max_date)
date_range = st.sidebar.date_input("Visit Date Range:", value=st.session_state["date_filter"], key="date_filter")

# Reset Filters button: delete keys then rerun using st.rerun()
if st.sidebar.button("Reset Filters"):
    for key in ["dept_filter", "gender_filter", "phys_filter", "age_filter", "date_filter"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Filter the DataFrame
df_filtered = df.copy()
if dept_selection:
    df_filtered = df_filtered[df_filtered["department"].isin(dept_selection)]
if gender_selection:
    df_filtered = df_filtered[df_filtered["gender"].isin(gender_selection)]
if phys_selection:
    df_filtered = df_filtered[df_filtered["physician_id"].isin(phys_selection)]
if age_group_selection:
    df_filtered = df_filtered[df_filtered["age_group"].isin(age_group_selection)]
start_date, end_date = date_range
df_filtered = df_filtered[(df_filtered["visit_date"] >= pd.to_datetime(start_date)) &
                          (df_filtered["visit_date"] <= pd.to_datetime(end_date))]

# ------------------------------
# Function for Keyword Extraction
# ------------------------------
def get_top_keywords(text, num=10):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(filtered_words).most_common(num)

# ------------------------------
# Lazy Load NLP Models in the Text Analysis Tab
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_summarization_pipeline():
    summarizer = pipeline("summarization", model="t5-small")
    return summarizer

@st.cache_resource(show_spinner=True)
def load_sentiment_pipeline():
    sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return sentiment

# ------------------------------
# Additional Topic Modeling Function (LDA)
# ------------------------------
@st.cache_data(show_spinner=True)
def perform_topic_modeling(text, n_topics=3, n_top_words=5):
    # Convert text to TF-IDF matrix using our stopwords list (list type)
    vectorizer = TfidfVectorizer(stop_words=list(stopwords))
    tfidf = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        topics[f"Topic {topic_idx+1}"] = [feature_names[i] for i in top_indices]
    return topics

# ------------------------------
# Set Up Tabs: Overview EDA, Text Analysis, Lab Trends & Clustering
# ------------------------------
tab_overview, tab_text, tab_lab = st.tabs(["Overview EDA", "Text Analysis", "Lab Trends & Clustering"])

# ------------------------------------------
# Tab 1: Overview EDA with Plotly (Enhanced Breakdown Analysis)
# ------------------------------------------
with tab_overview:
    st.header("Overview EDA")
    st.markdown("Interactive charts below provide segmented views and breakdowns for deeper insights into the synthetic EHR data.")
    
    # Age Distribution by Gender
    st.subheader("Age Distribution by Gender")
    fig_age_gender = px.histogram(df_filtered, x="age", color="gender", nbins=30,
                                  title="Age Distribution by Gender")
    fig_age_gender.update_layout(xaxis_title="Age", yaxis_title="Number of Visits")
    st.plotly_chart(fig_age_gender, use_container_width=True)
    st.caption("Histogram of patient ages segmented by gender to identify differences between male and female age profiles.")
    
    # Department Distribution by Age Group (Stacked Bar)
    st.subheader("Department Distribution by Age Group")
    dept_age = df_filtered.groupby(["department", "age_group"]).size().reset_index(name="visits")
    fig_dept_age = px.bar(dept_age, x="department", y="visits", color="age_group",
                          title="Visits by Department and Age Group", text="visits", barmode="stack")
    fig_dept_age.update_layout(xaxis_title="Department", yaxis_title="Number of Visits")
    st.plotly_chart(fig_dept_age, use_container_width=True)
    st.caption("Stacked bar chart showing how visit distribution within each department varies across age groups.")

    # Monthly Visit Trends by Department
    st.subheader("Monthly Visit Trends by Department")
    trend_dept = df_filtered.groupby([pd.Grouper(key="visit_date", freq="M"), "department"]).size().reset_index(name="visits")
    fig_trend_dept = px.line(trend_dept, x="visit_date", y="visits", color="department",
                             title="Monthly Visit Trends by Department", markers=True)
    fig_trend_dept.update_layout(xaxis_title="Month", yaxis_title="Number of Visits")
    st.plotly_chart(fig_trend_dept, use_container_width=True)
    st.caption("Line chart showing monthly visit counts for each department, highlighting seasonal trends by department.")

    # Laboratory Value Distributions by Age Group
    st.subheader("Laboratory Value Distributions by Age Group")
    col1, col2 = st.columns(2)
    with col1:
        fig_glucose = px.histogram(df_filtered, x="blood_glucose_level", nbins=30, color="age_group",
                                   title="Blood Glucose by Age Group",
                                   color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_glucose.update_layout(xaxis_title="Blood Glucose (mg/dL)", yaxis_title="Count")
        st.plotly_chart(fig_glucose, use_container_width=True)
        st.caption("Histogram of blood glucose levels segmented by age group to reveal differences across life stages.")
    with col2:
        fig_chol = px.histogram(df_filtered, x="cholesterol_level", nbins=30, color="age_group",
                                title="Cholesterol by Age Group",
                                color_discrete_sequence=px.colors.qualitative.Plotly)
        fig_chol.update_layout(xaxis_title="Cholesterol (mg/dL)", yaxis_title="Count")
        st.plotly_chart(fig_chol, use_container_width=True)
        st.caption("Histogram of cholesterol levels segmented by age group, useful to identify risk groups.")

    # Correlation Heatmap using Plotly
    st.subheader("Correlation Matrix")
    num_cols = ["age", "blood_glucose_level", "cholesterol_level", "num_diagnoses", "num_medications"]
    # Calculate numerical columns if missing
    if "num_diagnoses" not in df_filtered.columns:
        df_filtered["num_diagnoses"] = df_filtered["ICD10_codes"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    if "num_medications" not in df_filtered.columns:
        df_filtered["num_medications"] = df_filtered["medications"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    corr_matrix = df_filtered[num_cols].corr().round(2)
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         title="Correlation Matrix",
                         labels={"x": "Variable", "y": "Variable", "color": "Correlation"})
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("The correlation matrix shows relationships between patient age, lab values, and counts of diagnoses and medications.")

# ------------------------------------------
# Tab 2: Text Analysis with Enhanced NLP and Topic Modeling
# ------------------------------------------
with tab_text:
    st.header("Text Analysis")
    st.markdown("This section covers NLP analysis on clinical text. It includes keyword extraction, word clouds, sentiment analysis, summarization, and topic modeling using LDA.")
    
    summarizer = load_summarization_pipeline()
    sentiment_pipeline = load_sentiment_pipeline()
    
    # Keyword Extraction for Clinician Notes
    st.subheader("Top Keywords in Clinician Notes")
    all_notes_text = " ".join(df_filtered["clinician_note"].dropna().tolist()).lower()
    note_keywords = get_top_keywords(all_notes_text, num=10)
    st.write("**Top 10 Keywords in Clinician Notes:**")
    if note_keywords:
        for word, freq in note_keywords:
            st.write(f"- **{word}**: {freq}")
    else:
        st.write("No clinician notes available.")
    st.caption("Keywords extracted after exhaustive stopword removal to highlight informative terms from clinician notes.")

    # Keyword Extraction for Patient Feedback
    st.subheader("Top Keywords in Patient Feedback")
    all_feedback_text = " ".join(df_filtered["patient_feedback"].dropna().tolist()).lower()
    feedback_keywords = get_top_keywords(all_feedback_text, num=10)
    st.write("**Top 10 Keywords in Patient Feedback:**")
    if feedback_keywords:
        for word, freq in feedback_keywords:
            st.write(f"- **{word}**: {freq}")
    else:
        st.write("No patient feedback available.")
    st.caption("Keywords are extracted from patient feedback using an extensive stopword list to ensure meaningful terms remain.")

    # Topic Modeling on Clinician Notes with LDA
    st.subheader("Topic Modeling on Clinician Notes (LDA)")
    if all_notes_text.strip():
        topics = perform_topic_modeling(all_notes_text, n_topics=3, n_top_words=5)
        st.write("**Extracted Topics:**")
        for topic, words in topics.items():
            st.write(f"- **{topic}:** {', '.join(words)}")
        st.caption("LDA was applied on clinician notes (TF-IDF with exhaustive stopwords) to uncover dominant topics.")
    else:
        st.write("No clinician notes available for topic modeling.")
    
    # Word Clouds using Plotly
    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)
    with col1:
        if all_notes_text.strip():
            note_wc = WordCloud(width=400, height=300, stopwords=stopwords, background_color="white").generate(all_notes_text)
            fig_wc_notes = px.imshow(note_wc.to_array())
            fig_wc_notes.update_layout(coloraxis_showscale=False, title="Clinician Notes Word Cloud")
            st.plotly_chart(fig_wc_notes, use_container_width=True)
            st.caption("Word cloud for clinician notes after thorough stopword removal.")
        else:
            st.write("No clinician notes available.")
    with col2:
        if all_feedback_text.strip():
            fb_wc = WordCloud(width=400, height=300, stopwords=stopwords, background_color="white").generate(all_feedback_text)
            fig_wc_fb = px.imshow(fb_wc.to_array())
            fig_wc_fb.update_layout(coloraxis_showscale=False, title="Patient Feedback Word Cloud")
            st.plotly_chart(fig_wc_fb, use_container_width=True)
            st.caption("Word cloud for patient feedback highlighting salient terms.")
        else:
            st.write("No patient feedback available.")
    
    # Sentiment Analysis on Patient Feedback
    st.subheader("Patient Feedback Sentiment Analysis")
    if len(df_filtered) > 0 and df_filtered["patient_feedback"].notna().sum() > 0:
        sample_feedback = df_filtered["patient_feedback"].dropna().sample(min(100, df_filtered["patient_feedback"].dropna().shape[0]), random_state=42).tolist()
        sentiment_results = load_sentiment_pipeline()(sample_feedback)
        sentiments = []
        for result in sentiment_results:
            label = result.get("label", "")
            if "4" in label or "5" in label or label.upper() == "POSITIVE":
                sentiments.append("Positive")
            elif "1" in label or "2" in label or label.upper() == "NEGATIVE":
                sentiments.append("Negative")
            else:
                sentiments.append("Neutral")
        sent_counts = Counter(sentiments)
        sent_df = pd.DataFrame(list(sent_counts.items()), columns=["Sentiment", "Count"])
        fig_sent = px.bar(sent_df, x="Sentiment", y="Count", color="Sentiment",
                          title="Patient Feedback Sentiment Distribution", text="Count")
        fig_sent.update_layout(xaxis_title="Sentiment", yaxis_title="Count")
        st.plotly_chart(fig_sent, use_container_width=True)
        st.caption("This chart shows the distribution of sentiment in patient feedback, as determined by a BERT-based model.")
    else:
        st.write("No patient feedback available for sentiment analysis.")
    
    # Clinician Note Summarization
    st.subheader("Clinician Note Summarization")
    if df_filtered.shape[0] > 0 and df_filtered["clinician_note"].notna().any():
        sample_idx = st.number_input("Select a record index to summarize:", min_value=0, max_value=len(df_filtered)-1, value=0, step=1)
        sample_note = df_filtered.iloc[int(sample_idx)]["clinician_note"]
        st.write("**Original Clinician Note:**")
        st.write(sample_note)
        summary = summarizer(sample_note, max_length=60, min_length=20, do_sample=False)
        summary_text = summary[0]["summary_text"] if summary and len(summary) > 0 else "(No summary available)"
        st.write("**Summarized Note:**")
        st.write(summary_text)
        st.caption("The T5-small summarization model condenses the clinician note into a shorter, concise summary.")
    else:
        st.write("No clinician note available for summarization.")

# ------------------------------------------
# Tab 3: Lab Trends & Clustering with Enhanced Analysis
# ------------------------------------------
with tab_lab:
    st.header("Lab Trends & Patient Clustering")
    st.markdown("This section includes time-series analysis of lab values and unsupervised clustering of patient visits with additional evaluations.")

    # Monthly Lab Trends
    st.subheader("Monthly Average Lab Values")
    if len(df_filtered) > 0:
        lab_trends = df_filtered.groupby(pd.Grouper(key="visit_date", freq="M"))[["blood_glucose_level", "cholesterol_level"]].mean().reset_index()
        fig_glucose = px.line(lab_trends, x="visit_date", y="blood_glucose_level", title="Avg Blood Glucose Over Time", markers=True, color_discrete_sequence=["red"])
        fig_glucose.update_layout(xaxis_title="Month", yaxis_title="Avg Blood Glucose (mg/dL)")
        st.plotly_chart(fig_glucose, use_container_width=True)
        st.caption("The red line shows the monthly average blood glucose levels. Peaks may suggest a high prevalence of diabetes during those months.")

        fig_chol = px.line(lab_trends, x="visit_date", y="cholesterol_level", title="Avg Cholesterol Over Time", markers=True, color_discrete_sequence=["purple"])
        fig_chol.update_layout(xaxis_title="Month", yaxis_title="Avg Cholesterol (mg/dL)")
        st.plotly_chart(fig_chol, use_container_width=True)
        st.caption("The purple line represents the monthly average cholesterol levels, highlighting potential hyperlipidemia trends.")
    else:
        st.write("No lab data available for the selected filters.")

    # KMeans Clustering of Visits with Silhouette Score Calculation
    st.subheader("KMeans Clustering of Visits")
    features = ["age", "blood_glucose_level", "cholesterol_level", "num_diagnoses"]
    if "num_diagnoses" not in df_filtered.columns:
        df_filtered["num_diagnoses"] = df_filtered["ICD10_codes"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    if "num_medications" not in df_filtered.columns:
        df_filtered["num_medications"] = df_filtered["medications"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    if len(df_filtered) > 0:
        X = df_filtered[features].dropna().astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0)
        cluster_labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, cluster_labels)
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(X_scaled)
        cluster_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": cluster_labels.astype(str)
        })
        fig_cluster = px.scatter(cluster_df, x="PC1", y="PC2", color="Cluster",
                                 title="Patient Visit Clusters (PCA Projection)",
                                 labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"})
        st.plotly_chart(fig_cluster, use_container_width=True)
        st.caption(f"PCA scatter plot of KMeans clusters (Silhouette Score: {sil_score:.2f}). Distinct clusters suggest varying patient profiles (for example, older patients with high lab values versus younger patients with normal readings).")
        
        centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
        st.write("**Cluster Centers (approximate original scale):**")
        st.dataframe(centers.style.format("{:.1f}"))
        
        # Additional Analysis: Breakdown of Clusters by Age Group
        df_filtered["Cluster"] = cluster_labels.astype(str)
        cluster_age = df_filtered.groupby(["Cluster", "age_group"]).size().reset_index(name="visits")
        fig_cluster_age = px.bar(cluster_age, x="Cluster", y="visits", color="age_group",
                                 title="Age Group Distribution by Cluster", text="visits", barmode="stack")
        fig_cluster_age.update_layout(xaxis_title="Cluster", yaxis_title="Number of Visits")
        st.plotly_chart(fig_cluster_age, use_container_width=True)
        st.caption("Stacked bar chart showing the age group breakdown within each cluster, providing insight into the demographics of each group.")
    else:
        st.write("Not enough data available for clustering.")
