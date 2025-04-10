# 🏥 EHR Visits Analysis & NLP Pipeline

## Overview
The **ehr_visits** dataset is a rich collection of electronic health record (EHR) data, encompassing:
- **Demographics:** Patient age, gender, and more.
- **Clinical Data:** Departments, physician IDs, ICD-10 diagnosis codes, medications, lab results.
- **Textual Notes:** Clinician notes and patient feedback.

Our goal is to perform a comprehensive **Exploratory Data Analysis (EDA)** and build an **NLP pipeline** to extract actionable insights. The results are shared through an interactive Streamlit dashboard and a GitHub Pages report.

> **Why EDA & NLP?**  
> EDA uncovers hidden patterns and relationships to inform better patient care, while NLP transforms unstructured text into valuable insights on sentiment, topics, and summarization.

---

## 📊 Exploratory Data Analysis (EDA)
Our EDA dives deep into the dataset with the following focus areas:

- **Demographics:**  
  - Analyzing patient age and gender distributions.
  - Grouping patients into age categories.
  - Tracking trends over time.

- **Visit Trends:**  
  - Analyzing visits by department and physician.
  - Uncovering seasonal patterns via monthly trends.

- **Diagnosis & Medications:**  
  - Evaluating ICD-10 codes frequency.
  - Identifying common chronic conditions (e.g., diabetes, hypertension, hyperlipidemia).
  - Reviewing prescribed medications.

- **Lab Results:**  
  - Analyzing distributions of blood glucose and cholesterol.
  - Implementing cutoff thresholds to flag potential risk cases.

- **Correlation Analysis:**  
  - Constructing a correlation matrix to explore relationships among numerical features (age, lab values, the number of diagnoses, medications).

---

## 🧠 NLP Analysis
The NLP pipeline processes unstructured text with cutting-edge techniques:

- **Preprocess Text:**  
  - Extensive stopword removal to clean clinician notes and patient feedback.

- **Keyword Extraction:**  
  - Extracting top informative keywords from the text.

- **Topic Modeling:**  
  - Applying Latent Dirichlet Allocation (LDA) to clinician notes to uncover dominant themes.

- **Sentiment Analysis:**  
  - Using a BERT-based model to classify patient feedback sentiment.

- **Text Summarization:**  
  - Employing a T5 model to condense lengthy notes into concise summaries.

---

## 📈 Lab Trends & Clustering
Additional analysis includes:

- **Time-Series Trends:**  
  - Monitoring monthly averages for key lab values (e.g., blood glucose, cholesterol) to observe health trends.

- **Unsupervised Clustering:**  
  - Using KMeans clustering (validated with silhouette scores) to segment visits into distinct clusters.
  - Visualizing clusters using PCA projections.
  - Analyzing demographic breakdowns within each cluster.

---

## 📁 Repository Structure
- **`data/synthetic_ehr_data.csv`**: Synthetic dataset containing 10,000 visit records.
- **`streamlit_app.py`**: Source code for the interactive Streamlit dashboard.
- **`requirements.txt`**: Necessary installation packages before initializing this app.
- **`README.md`**: This file.

**Optional Files:**
- Additional Markdown files (e.g., `EDA.md`, `NLP.md`, `Clustering.md`) for the GitHub Pages report.
- Custom model directories (if using custom trained models).

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your_username/ehr-visits-nlp-pipeline.git
cd ehr-visits-nlp-pipeline
```

### 2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 3. **Generate Synthetic Data:**
   ```bash
   python data/generate_synthetic_ehr_data.py
   ```
   *This command creates `data/synthetic_ehr_data.csv`.*

### 4. Launch the Streamlit App
```bash
streamlit run main.py
```

## 🔗 Live Demo

👉 [Try the live app here](https://ehr-visits-nlp-pipeline-nwstsqsytx9abjoe7tpjus.streamlit.app/)

---

## 🌐 Deploying on GitHub Pages

1. **Create Documentation Folder:**  
   - Create a `docs/` folder with your additional Markdown report files.

2. **Setup MkDocs:**  
   - Add an `mkdocs.yml` file (refer to the sample configuration).

3. **Install MkDocs & Material Theme:**
   ```bash
   pip install mkdocs mkdocs-material
   ```

4. **Serve Locally:**
   ```bash
   mkdocs serve
   ```

5. **Deploy:**
   ```bash
   mkdocs gh-deploy
   ```
