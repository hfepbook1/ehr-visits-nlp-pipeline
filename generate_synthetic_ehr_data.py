import pandas as pd
import numpy as np
import random
from datetime import datetime, date, timedelta
import calendar

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define possible departments (specialties)
departments = ["Cardiology", "Neurology", "Orthopedics", "General Medicine",
               "Endocrinology", "Pulmonology", "Psychiatry", "Oncology", "Pediatrics"]

# Create a mapping of physicians to departments (5 physicians per department for example)
physicians_by_dept = {}
physician_id_to_dept = {}
phys_id = 1
for dept in departments:
    physicians_by_dept[dept] = list(range(phys_id, phys_id+5))
    for pid in physicians_by_dept[dept]:
        physician_id_to_dept[pid] = dept
    phys_id += 5

# Define ICD-10 code clusters for comorbidities (diagnosis groups)
clusters = [
    {"name": "Metabolic", "codes": ["I10", "E11.9", "E78.5"], "dept": "General Medicine"},   # Hypertension, Type 2 Diabetes, Hyperlipidemia
    {"name": "Cardiac", "codes": ["I25.10", "I50.9", "I10"], "dept": "Cardiology"},         # CAD, Heart Failure (+ maybe HTN)
    {"name": "Respiratory", "codes": ["J45.909", "J44.9"], "dept": "Pulmonology"},          # Asthma, COPD
    {"name": "Mental Health", "codes": ["F32.9", "F41.9"], "dept": "Psychiatry"},          # Depression, Anxiety
    {"name": "Neurology", "codes": ["G43.909"], "dept": "Neurology"},                     # Migraine
    {"name": "Neuro/Cardio", "codes": ["I63.9", "I10"], "dept": "Neurology"},             # Stroke + Hypertension
    {"name": "Orthopedic", "codes": ["M19.90", "M54.5"], "dept": "Orthopedics"},          # Osteoarthritis, Low back pain
    {"name": "Endocrine", "codes": ["E03.9"], "dept": "Endocrinology"},                  # Hypothyroidism
    {"name": "Oncology", "codes": ["C50.9"], "dept": "Oncology"},                       # Breast cancer
    {"name": "General Acute", "codes": ["J06.9"], "dept": "General Medicine"}           # Upper respiratory infection (common cold)
]
# Pediatric-specific clusters for patients under 18 (e.g. colds and asthma)
pediatric_clusters = [
    {"name": "Peds Cold", "codes": ["J06.9"], "dept": "Pediatrics"},     # URI in children
    {"name": "Peds Asthma", "codes": ["J45.909"], "dept": "Pediatrics"}  # Asthma in children
]

# Define age group categories for later use
age_group_labels = ["0-17", "18-34", "35-49", "50-69", "70+"]

# Decide number of patients and visits
num_patients = 8000
total_visits = 10000
patient_ids = list(range(1, num_patients+1))

# Generate base demographics for patients
ages = []    # age for each patient
genders = [] # gender for each patient
for pid in patient_ids:
    # Assign age by group to ensure a mix of age ranges
    # We'll give roughly equal distribution across our defined age groups
    age_group_choice = random.choices(age_group_labels, weights=[0.2,0.2,0.2,0.2,0.2], k=1)[0]
    if age_group_choice == "0-17":
        age = random.randint(0, 17)
    elif age_group_choice == "18-34":
        age = random.randint(18, 34)
    elif age_group_choice == "35-49":
        age = random.randint(35, 49)
    elif age_group_choice == "50-69":
        age = random.randint(50, 69)
    else:
        age = random.randint(70, 90)
    ages.append(age)
    # Assign gender randomly
    genders.append(random.choice(["Male", "Female"]))

# Assign each patient a diagnosis cluster (with comorbidities) based on age
patient_cluster = []  # will store a tuple (cluster_type, index) for each patient
for pid, age in zip(patient_ids, ages):
    if age < 18:
        # Pediatric patient: choose from pediatric clusters
        idx = random.choices(range(len(pediatric_clusters)), weights=[0.7, 0.3], k=1)[0]
        patient_cluster.append(("pediatric", idx))
    else:
        # Adult patient: choose from adult clusters with predefined probabilities
        idx = random.choices(range(len(clusters)), 
                              weights=[0.18, 0.10, 0.10, 0.10, 0.05, 0.07, 0.10, 0.05, 0.05, 0.20], k=1)[0]
        patient_cluster.append(("adult", idx))

# Determine actual ICD-10 codes for each patient based on their cluster
patient_diagnoses = []  # list of lists of codes for each patient
for (ptype, c_idx) in patient_cluster:
    if ptype == "adult":
        code_list = clusters[c_idx]["codes"]
    else:
        code_list = pediatric_clusters[c_idx]["codes"]
    # For clusters with multiple codes, randomly decide how many of those codes the patient has
    if len(code_list) == 1:
        chosen_codes = code_list[:]  # single code cluster
    elif len(code_list) == 2:
        # 60% chance the patient has both diagnoses, 40% chance only one
        if random.random() < 0.6:
            chosen_codes = code_list[:]
        else:
            chosen_codes = [random.choice(code_list)]
    else:
        # Cluster with 3 codes (e.g., metabolic): give patient 2 or all 3 (to reflect comorbidity)
        if random.random() < 0.5:
            chosen_codes = random.sample(code_list, 2)  # two of the three
        else:
            chosen_codes = code_list[:]
    patient_diagnoses.append(chosen_codes)

# We will now generate visit records. Some patients will have a second visit to reach 10,000 visits total.
# Determine which patients get an extra visit
extra_visits_count = total_visits - num_patients  # number of additional visits beyond first visits
extra_visit_patients = set(random.sample(patient_ids, extra_visits_count))

visit_records = []
visit_id = 1
first_visit_date = {}  # store each patient's first visit date to schedule second one after it

# Helper: mapping from ICD-10 to medication names
med_map = {
    "I10": "Lisinopril",       # hypertension -> Lisinopril
    "E11.9": "Metformin",      # type 2 diabetes -> Metformin
    "E78.5": "Atorvastatin",   # hyperlipidemia -> Atorvastatin
    "I25.10": "Aspirin",       # CAD -> Aspirin (as prevention)
    "I50.9": "Furosemide",     # heart failure -> Furosemide
    "J45.909": "Albuterol",    # asthma -> Albuterol inhaler
    "J44.9": "Tiotropium",     # COPD -> Tiotropium inhaler
    "F32.9": "Sertraline",     # depression -> Sertraline
    "F41.9": "Alprazolam",     # anxiety -> Alprazolam
    "G43.909": "Sumatriptan",  # migraine -> Sumatriptan
    "I63.9": "Clopidogrel",    # stroke -> Clopidogrel
    "M19.90": "Ibuprofen",     # osteoarthritis -> Ibuprofen
    "M54.5": "Naproxen",       # back pain -> Naproxen
    "E03.9": "Levothyroxine",  # hypothyroidism -> Levothyroxine
    "C50.9": "Tamoxifen",      # breast cancer -> Tamoxifen
    "J06.9": "Acetaminophen"   # URI (cold) -> Acetaminophen
}

# Predefined mapping of ICD codes to descriptive text for notes
code_desc = {
    "I10": "hypertension",
    "E11.9": "type 2 diabetes",
    "E78.5": "hyperlipidemia",
    "I25.10": "coronary artery disease",
    "I50.9": "heart failure",
    "J45.909": "asthma",
    "J44.9": "COPD",
    "F32.9": "depression",
    "F41.9": "anxiety",
    "G43.909": "migraine",
    "I63.9": "stroke",
    "M19.90": "osteoarthritis",
    "M54.5": "low back pain",
    "E03.9": "hypothyroidism",
    "C50.9": "breast cancer",
    "J06.9": "upper respiratory infection"
}

# Define month weights for seasonality (more visits in winter, fewer in summer)
month_weights = [0.11, 0.10, 0.07, 0.06, 0.07, 0.06, 0.07, 0.07, 0.09, 0.11, 0.10, 0.09]  # Jan..Dec
# Define year weights (to simulate slight increase each year)
year_weights = [0.15, 0.18, 0.20, 0.22, 0.25]  # for years 2020-2024

# Function to generate a random date (optionally after a given date) 
def random_date(year_start=2020, year_end=2024, after_date=None):
    if after_date is None:
        # pick a year with weighted distribution
        year = random.choices([2020, 2021, 2022, 2023, 2024], weights=year_weights, k=1)[0]
        month = random.choices(range(1, 13), weights=month_weights, k=1)[0]
        # Choose day within that month/year
        _, last_day = calendar.monthrange(year, month)
        day = random.randint(1, last_day)
        return date(year, month, day)
    else:
        # generate a date after a given date (for second visit)
        start_date = after_date
        end_date = date(2024, 12, 31)
        if start_date >= end_date:
            return start_date  # no later date available, use the same day
        # pick random days offset between 0 and (end_date - start_date)
        delta_days = (end_date - start_date).days
        offset = random.randint(0, delta_days)
        return start_date + timedelta(days=offset)

# Generate records for each patient's first visit
for pid, age, gender, (ptype, cluster_idx), diag_codes in zip(patient_ids, ages, genders, patient_cluster, patient_diagnoses):
    # Determine visit date for first visit
    visit_date = random_date()
    first_visit_date[pid] = visit_date
    # Determine department: Pediatrics for under 18, otherwise from cluster mapping
    if age < 18:
        dept = "Pediatrics"
    else:
        dept = clusters[cluster_idx]["dept"] if ptype == "adult" else pediatric_clusters[cluster_idx]["dept"]
    # Assign a physician from that department
    physician_id = random.choice(physicians_by_dept[dept])
    # Determine lab values (blood_glucose and cholesterol) tied to diagnoses
    # Start with normal baseline, then adjust if certain diagnoses present
    blood_glucose = random.randint(70, 110)  # normal baseline (mg/dL)
    cholesterol = random.randint(150, 230)   # normal baseline (mg/dL)
    if any(code.startswith("E11") or code.startswith("E10") for code in diag_codes):
        # If diabetes present, blood glucose tends to be higher
        blood_glucose = random.randint(130, 250)
    if any(code.startswith("E78") or code.startswith("I25") for code in diag_codes):
        # If hyperlipidemia or CAD present, cholesterol tends to be higher
        cholesterol = random.randint(240, 320)
    # Build clinician note (3 sentences: history, current visit, plan)
    chronic_list = [code_desc[c] for c in diag_codes if c in code_desc and c not in ["J06.9"]]
    # (We'll treat J06.9 as acute only, not chronic history)
    acute_list = [code_desc[c] for c in diag_codes if c in code_desc and c in ["J06.9", "G43.909", "stroke", "migraine"]]
    # Determine if patient has chronic conditions to mention
    if chronic_list:
        if len(chronic_list) == 1:
            hist_text = f"a history of {chronic_list[0]}"
        else:
            # combine multiple with commas and 'and'
            hist_text = "a history of " + ", ".join(chronic_list[:-1]) + " and " + chronic_list[-1]
    else:
        hist_text = "no significant past medical history"
    # Demographics
    dem_text = f"Patient is a {age}-year-old {'male' if gender=='Male' else 'female'} with {hist_text}."
    # Presenting text
    if acute_list:
        present_text = f"{'He' if gender=='Male' else 'She'} presents with {acute_list[0]}."  # e.g., "presents with migraine."
    else:
        # If no acute issue, assume follow-up of first chronic condition or general check
        if chronic_list:
            present_text = f"{'He' if gender=='Male' else 'She'} comes in for follow-up of {chronic_list[0]}."
        else:
            present_text = f"{'He' if gender=='Male' else 'She'} is here for a routine check-up."
    # Plan text
    plan_text = "Plan to continue current medications and follow up in 3 months."
    if any("diabetes" in desc for desc in chronic_list):
        plan_text = "Plan to check HbA1c and adjust medications accordingly."
    elif acute_list:
        if "infection" in acute_list[0] or "respiratory" in acute_list[0]:
            plan_text = "Advised rest and hydration."
        elif "pain" in acute_list[0] or "migraine" in acute_list[0]:
            plan_text = "Advised to take pain relievers as needed."
    clinician_note = f"{dem_text} {present_text} {plan_text}"
    # Generate patient feedback with a sentiment
    sentiment = random.choices(["Positive", "Neutral", "Negative"], weights=[0.6, 0.3, 0.1], k=1)[0]
    if sentiment == "Positive":
        feedback_options = [
            "The doctor was very helpful and answered all my questions.",
            "I am feeling much better after the visit.",
            "Great service and friendly staff.",
            "My experience was very positive overall."
        ]
    elif sentiment == "Neutral":
        feedback_options = [
            "The visit was okay, nothing special to mention.",
            "It was a routine visit and everything was fine.",
            "The appointment was fine and my concerns were addressed.",
            "I have no strong feelings; it went as expected."
        ]
    else:  # Negative
        feedback_options = [
            "I had to wait too long to see the doctor.",
            "I'm not satisfied with the explanation I received.",
            "The visit felt rushed and I left with unanswered questions.",
            "I am disappointed with the care I received today."
        ]
    patient_feedback = random.choice(feedback_options)
    # Determine medications for this visit (based on diagnoses)
    meds = []
    for code in diag_codes:
        if code in med_map:
            med = med_map[code]
            if med not in meds:
                meds.append(med)
    # Create the visit record dictionary
    record = {
        "patient_id": pid,
        "visit_id": visit_id,
        "visit_date": visit_date,
        "age": age,
        "gender": gender,
        "department": dept,
        "physician_id": physician_id,
        "ICD10_codes": diag_codes,
        "medications": meds,
        "blood_glucose_level": blood_glucose,
        "cholesterol_level": cholesterol,
        "clinician_note": clinician_note,
        "patient_feedback": patient_feedback,
        "age_group": None,       # to fill later
        "visit_season": None,    # to fill later
        "num_diagnoses": len(diag_codes),
        "num_medications": len(meds),
        "has_diabetes": any(code.startswith("E11") or code.startswith("E10") for code in diag_codes),
        "has_hypertension": any(code == "I10" for code in diag_codes),
        "has_hyperlipidemia": any(code.startswith("E78") for code in diag_codes),
        "has_heart_disease": any(code.startswith("I25") or code.startswith("I50") for code in diag_codes),
        "has_respiratory": any(code.startswith("J45") or code.startswith("J44") for code in diag_codes),
        "has_mental": any(code.startswith("F32") or code.startswith("F41") for code in diag_codes),
        "feedback_sentiment": sentiment
    }
    visit_records.append(record)
    visit_id += 1

# Generate second visit for patients selected for an extra visit
for pid in extra_visit_patients:
    # Use the patient's first visit date to ensure the second visit is on or after it
    first_date = first_visit_date[pid]
    second_date = random_date(after_date=first_date)
    # Patient's base info
    idx = pid - 1  # index in arrays (since patient_ids are 1-indexed)
    age_at_first = ages[idx]
    gender = genders[idx]
    diag_codes = patient_diagnoses[idx]
    # Age at second visit (add year difference if any)
    age_at_second = age_at_first + (second_date.year - first_date.year)
    # Determine department (same logic as first visit)
    if age_at_second < 18:
        dept = "Pediatrics"
    else:
        cl_type, cl_idx = patient_cluster[idx]
        dept = clusters[cl_idx]["dept"] if cl_type == "adult" else pediatric_clusters[cl_idx]["dept"]
    physician_id = random.choice(physicians_by_dept[dept])
    # Labs for second visit
    blood_glucose = random.randint(70, 110)
    cholesterol = random.randint(150, 230)
    if any(code.startswith("E11") or code.startswith("E10") for code in diag_codes):
        blood_glucose = random.randint(130, 250)
    if any(code.startswith("E78") or code.startswith("I25") for code in diag_codes):
        cholesterol = random.randint(240, 320)
    # Clinician note for second visit (may be similar since chronic conditions persist)
    chronic_list = [code_desc[c] for c in diag_codes if c in code_desc and c not in ["J06.9"]]
    acute_list = [code_desc[c] for c in diag_codes if c in code_desc and c in ["J06.9", "G43.909", "stroke", "migraine"]]
    if chronic_list:
        if len(chronic_list) == 1:
            hist_text = f"a history of {chronic_list[0]}"
        else:
            hist_text = "a history of " + ", ".join(chronic_list[:-1]) + " and " + chronic_list[-1]
    else:
        hist_text = "no significant past medical history"
    dem_text = f"Patient is a {age_at_second}-year-old {'male' if gender=='Male' else 'female'} with {hist_text}."
    if acute_list:
        present_text = f"{'He' if gender=='Male' else 'She'} presents with {acute_list[0]}."
    else:
        if chronic_list:
            present_text = f"{'He' if gender=='Male' else 'She'} comes in for follow-up of {chronic_list[0]}."
        else:
            present_text = f"{'He' if gender=='Male' else 'She'} is here for a routine check-up."
    plan_text = "Plan to continue current medications and follow up in 3 months."
    if any("diabetes" in desc for desc in chronic_list):
        plan_text = "Plan to check HbA1c and adjust medications accordingly."
    elif acute_list:
        if "infection" in acute_list[0] or "respiratory" in acute_list[0]:
            plan_text = "Advised rest and hydration."
        elif "pain" in acute_list[0] or "migraine" in acute_list[0]:
            plan_text = "Advised to take pain relievers as needed."
    clinician_note = f"{dem_text} {present_text} {plan_text}"
    sentiment = random.choices(["Positive", "Neutral", "Negative"], weights=[0.6, 0.3, 0.1], k=1)[0]
    if sentiment == "Positive":
        feedback_options = [
            "The doctor was very helpful and answered all my questions.",
            "I am feeling much better after the visit.",
            "Great service and friendly staff.",
            "My experience was very positive overall."
        ]
    elif sentiment == "Neutral":
        feedback_options = [
            "The visit was okay, nothing special to mention.",
            "It was a routine visit and everything was fine.",
            "The appointment was fine and my concerns were addressed.",
            "I have no strong feelings; it went as expected."
        ]
    else:
        feedback_options = [
            "I had to wait too long to see the doctor.",
            "I'm not satisfied with the explanation I received.",
            "The visit felt rushed and I left with unanswered questions.",
            "I am disappointed with the care I received today."
        ]
    patient_feedback = random.choice(feedback_options)
    meds = []
    for code in diag_codes:
        if code in med_map:
            med = med_map[code]
            if med not in meds:
                meds.append(med)
    record = {
        "patient_id": pid,
        "visit_id": visit_id,
        "visit_date": second_date,
        "age": age_at_second,
        "gender": gender,
        "department": dept,
        "physician_id": physician_id,
        "ICD10_codes": diag_codes,
        "medications": meds,
        "blood_glucose_level": blood_glucose,
        "cholesterol_level": cholesterol,
        "clinician_note": clinician_note,
        "patient_feedback": patient_feedback,
        "age_group": None,
        "visit_season": None,
        "num_diagnoses": len(diag_codes),
        "num_medications": len(meds),
        "has_diabetes": any(code.startswith("E11") or code.startswith("E10") for code in diag_codes),
        "has_hypertension": any(code == "I10" for code in diag_codes),
        "has_hyperlipidemia": any(code.startswith("E78") for code in diag_codes),
        "has_heart_disease": any(code.startswith("I25") or code.startswith("I50") for code in diag_codes),
        "has_respiratory": any(code.startswith("J45") or code.startswith("J44") for code in diag_codes),
        "has_mental": any(code.startswith("F32") or code.startswith("F41") for code in diag_codes),
        "feedback_sentiment": sentiment
    }
    visit_records.append(record)
    visit_id += 1

# Fill age_group and visit_season for all records
for rec in visit_records:
    age = rec["age"]
    if age <= 17:
        rec["age_group"] = "0-17"
    elif age <= 34:
        rec["age_group"] = "18-34"
    elif age <= 49:
        rec["age_group"] = "35-49"
    elif age <= 69:
        rec["age_group"] = "50-69"
    else:
        rec["age_group"] = "70+"
    m = rec["visit_date"].month
    if m in [12, 1, 2]:
        rec["visit_season"] = "Winter"
    elif m in [3, 4, 5]:
        rec["visit_season"] = "Spring"
    elif m in [6, 7, 8]:
        rec["visit_season"] = "Summer"
    else:
        rec["visit_season"] = "Fall"

# Create DataFrame
df = pd.DataFrame(visit_records)
# Sort by visit_date (chronological order of visits)
df.sort_values("visit_date", inplace=True, ignore_index=True)
# Reassign visit_id after sorting, if needed (here we keep as unique identifier regardless of order)
# df['visit_id'] = range(1, len(df)+1)

# Save dataset to CSV
df.to_csv("synthetic_ehr_data.csv", index=False)
print("Synthetic EHR dataset created with", len(df), "records.")
