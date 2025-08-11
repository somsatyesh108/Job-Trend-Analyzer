
import streamlit as st
import pandas as pd
import re, math
from io import StringIO, BytesIO
from datetime import datetime
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Career Analytics: Job Trends & Skill Gap", layout="wide")

DEFAULT_SKILLS = [
    "Python","Pandas","NumPy","Scikit-learn","TensorFlow","PyTorch","Machine Learning","Deep Learning","NLP",
    "SQL","Data Visualization","Matplotlib","Seaborn","Tableau","Power BI","MLOps","Docker","Kubernetes","AWS","GCP","Azure",
    "FastAPI","CI/CD","Airflow","Feature Stores","Model Serving","Spark","Hadoop","Kafka","ETL","Data Warehousing","Snowflake",
    "Redshift","DBT","JavaScript","TypeScript","React","Node","Express","REST","GraphQL","PostgreSQL","MongoDB","Redis","Celery",
    "Django","Flask","Unit Testing","Git","A/B Testing","Statistics","Excel","Storytelling"
]

ROLE_KEYWORDS = {
    "Data Scientist": ["data scientist"],
    "ML Engineer": ["ml engineer","machine learning engineer"],
    "Data Engineer": ["data engineer"],
    "Full Stack Developer": ["full stack"],
    "Backend Developer": ["backend"],
    "Business Analyst": ["business analyst","data analyst"]
}

ROLE_SKILLS = {
    "Data Scientist": ["Python","Pandas","NumPy","Scikit-learn","TensorFlow","PyTorch","Machine Learning","Deep Learning","NLP","SQL","Data Visualization","Tableau","Power BI"],
    "ML Engineer": ["Python","TensorFlow","PyTorch","MLOps","Docker","Kubernetes","AWS","GCP","Azure","FastAPI","CI/CD","Airflow"],
    "Data Engineer": ["Python","SQL","Spark","Hadoop","Kafka","Airflow","ETL","AWS","GCP","Azure","Data Warehousing","Snowflake","Redshift","DBT"],
    "Full Stack Developer": ["JavaScript","TypeScript","React","Node","Express","REST","PostgreSQL","MongoDB","Docker","Git","CI/CD","AWS"],
    "Backend Developer": ["Python","Django","FastAPI","Flask","REST","GraphQL","PostgreSQL","Redis","Celery","Docker","Unit Testing"],
    "Business Analyst": ["SQL","Excel","Power BI","Tableau","Data Visualization","Storytelling","A/B Testing","Statistics","Python"]
}

def read_pdf(file) -> str:
    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return ""

def extract_skills(text, vocab=None):
    if vocab is None:
        vocab = DEFAULT_SKILLS
    norm = re.sub(r"[^a-zA-Z0-9#+./ ]", " ", text.lower())
    found = set()
    for skill in vocab:
        pat = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pat, norm):
            found.add(skill)
    return sorted(found)

def infer_role(title):
    t = title.lower()
    for role, keys in ROLE_KEYWORDS.items():
        if any(k in t for k in keys):
            return role
    return "Other"

@st.cache_data
def load_data(csv_file) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    # expected columns: job_id,title,company,city,salary_min,salary_max,currency,date_posted,description
    # make sure types are as expected
    if "date_posted" in df.columns:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    else:
        df["date_posted"] = pd.NaT
    df["role"] = df["title"].apply(infer_role)
    df["salary_avg"] = ((df.get("salary_min", 0).fillna(0) + df.get("salary_max", 0).fillna(0)) / 2).astype(float)
    df["skills_in_jd"] = df["description"].fillna("").apply(lambda t: ", ".join(extract_skills(t)))
    return df

st.title("📈 Career Analytics — Job Market Trends & Skill Gap Predictor")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload jobs CSV (optional). If not, a sample will be used.", type=["csv"])
if uploaded is None:
    st.info("Using bundled sample dataset (data/sample_jobs.csv).")
    data_path = "data/sample_jobs.csv"
else:
    data_path = uploaded

df = load_data(data_path)

st.sidebar.subheader("Filters")
roles = ["All"] + sorted(df["role"].unique().tolist())
pick_role = st.sidebar.selectbox("Role", roles)
min_date, max_date = df["date_posted"].min(), df["date_posted"].max()
if pd.isna(min_date) or pd.isna(max_date):
    min_date = pd.to_datetime("2024-01-01")
    max_date = pd.to_datetime("today")
date_range = st.sidebar.date_input("Date range", (min_date, max_date))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["date_posted"] >= start) & (df["date_posted"] <= end)]
if pick_role != "All":
    df = df[df["role"] == pick_role]

st.markdown("### Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jobs", len(df))
c2.metric("Cities", df["city"].nunique())
c3.metric("Avg Salary (INR)", f"{int(df['salary_avg'].dropna().mean()):,}" if df["salary_avg"].notna().any() else "—")
c4.metric("Roles", df["role"].nunique())

with st.expander("Preview Data", expanded=False):
    st.dataframe(df.head(20))

st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trends", "Salary heatmap", "City distribution", "Clusters", "Skill Gap (upload resume)"])

with tab1:
    st.subheader("Monthly Job Trend")
    if "date_posted" in df.columns and df["date_posted"].notna().any():
        trend = df.set_index("date_posted").resample("M").size().reset_index(name="count")
        fig = px.line(trend, x="date_posted", y="count", markers=True, title="Jobs per Month")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No dates available to compute trends.")

with tab2:
    st.subheader("Role-wise Salary Heatmap")
    heat = df.groupby(["role","city"])["salary_avg"].mean().reset_index()
    if not heat.empty:
        pivot = heat.pivot(index="role", columns="city", values="salary_avg").fillna(0)
        fig = px.imshow(pivot, aspect="auto", title="Mean Salary (INR) by Role & City")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough salary data to show heatmap.")

with tab3:
    st.subheader("City-wise Job Distribution")
    city_counts = df["city"].value_counts().reset_index()
    city_counts.columns = ["city","count"]
    fig = px.bar(city_counts, x="city", y="count", title="Jobs by City")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Cluster Job Types")
    texts = df["description"].fillna("")
    if len(texts) >= 10:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        X = vectorizer.fit_transform(texts)
        k = 5 if len(df) >= 50 else 3
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        dfc = df.copy()
        dfc["cluster"] = labels
        st.write("Top keywords by cluster:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(k):
            top = [terms[ind] for ind in order_centroids[i, :8]]
            st.markdown(f"**Cluster {i}** — {', '.join(top)}")
        st.dataframe(dfc[["title","city","role","salary_avg","cluster","skills_in_jd"]].head(50))
    else:
        st.info("Need at least 10 job descriptions to cluster.")

with tab5:
    st.subheader("Upload Your Resume to See Skill Gap")
    uploaded_resume = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf","txt"], key="resume")
    resume_text = ""
    if uploaded_resume is not None:
        if uploaded_resume.name.lower().endswith(".pdf"):
            resume_text = read_pdf(uploaded_resume)
        else:
            resume_text = uploaded_resume.read().decode("utf-8", errors="ignore")
    manual_text = st.text_area("Or paste resume text here", "")
    if manual_text.strip():
        resume_text = manual_text

    if resume_text:
        resume_skills = extract_skills(resume_text)
        st.write("**Skills detected in your resume:**", ", ".join(resume_skills) if resume_skills else "None detected")

        # Choose a target role
        role_for_gap = st.selectbox("Target Role", list(ROLE_SKILLS.keys()))
        must_have = set(ROLE_SKILLS[role_for_gap])
        gaps = sorted(list(must_have.difference(set(resume_skills))))
        st.markdown(f"### Missing skills for **{role_for_gap}**")
        if gaps:
            st.warning(", ".join(gaps))
        else:
            st.success("Great! Your resume covers most common skills for this role.")

        # Compare with selected job postings
        st.markdown("### Job-specific Gap (pick a job)")
        job_idx = st.selectbox("Select job", df.index, format_func=lambda i: f"{df.loc[i, 'title']} — {df.loc[i,'company']} ({df.loc[i,'city']})")
        job_text = df.loc[job_idx, "description"]
        job_skills = set(extract_skills(job_text))
        missing_vs_job = sorted(list(job_skills.difference(set(resume_skills))))
        st.write("**Skills requested in this JD:**", ", ".join(job_skills) if job_skills else "None found")
        if missing_vs_job:
            st.error("You are missing: " + ", ".join(missing_vs_job))
        else:
            st.success("You match the skills in this JD!")

    else:
        st.info("Upload a resume PDF/TXT or paste text to compute the skill gap.")
