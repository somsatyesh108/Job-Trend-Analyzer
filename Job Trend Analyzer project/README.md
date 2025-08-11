# Career Analytics — Job Market Trend Analyzer & Skill Gap Predictor

This is a beginner-friendly starter project you can run on your laptop. It includes:

- A **sample jobs dataset** (`data/sample_jobs.csv`)
- A **Streamlit dashboard** (`src/streamlit_app.py`) with:
  - Monthly job trend
  - Role-wise salary heatmap
  - City-wise job distribution
  - Job description clustering
  - **Resume upload** to compute **skill gap**

## 1) Install software (Windows)

1. Install **Python 3.11** from https://www.python.org/downloads/ (check “Add Python to PATH” during install).
2. Install **VS Code** from https://code.visualstudio.com/ and add the “Python” extension.
3. (Optional) Install **Git** from https://git-scm.com/downloads.

## 2) Create a virtual environment

Open **Command Prompt (CMD)** and run:

```bash
cd PATH_TO_THIS_FOLDER/career_analytics_starter
python -m venv .venv
.venv\Scripts\activate
```

## 3) Install dependencies

```bash
pip install --upgrade pip
pip install streamlit pandas scikit-learn pdfplumber plotly
```

## 4) Run the app

```bash
streamlit run src/streamlit_app.py
```

It will open in your browser. Use the sample data or upload your own CSV of job postings.

## 5) Bring your own data later (optional)

Your CSV should have columns similar to:

```
job_id,title,company,city,salary_min,salary_max,currency,date_posted,description
```

Dates should be in `YYYY-MM-DD` format. Salaries are optional.

## 6) Next steps (optional)
- Replace sample data with real data from Kaggle or your own scraping.
- Improve skill extractor using spaCy PhraseMatcher or large skill dictionaries.
- Add authentication and save user profiles.
- Export insights as PDF/CSV for your resume review sessions.
