# 🤖 Autonomous Data Analysis Agent

Upload any CSV file (or **many files, a ZIP of folders, or a whole folder path**) and the agent **decides what analysis to run, runs it, explains the
results in plain English, answers your questions and writes a PDF report** — with no manual setup.

## What it does

| Step | Module | What happens |
|---|---|---|
| 1. Clean | `data_cleaner.py` | Tidies column names, trims text, converts numbers stored as text, removes empty/duplicate rows, fills missing values |
| 2. Profile | `data_profiler.py` | Rows, columns, types, missing values, identifier columns |
| 3. Decide | `agent.py` | Chooses EDA / Machine Learning / Anomaly Detection **and explains why** |
| 4. Execute | `agent_executor.py` | Summary statistics, correlations, charts, Random Forest model, Isolation Forest anomalies |
| 5. Explain | `llm_service.py` | Writes the insights in plain English |
| 6. Ask | `question_answering.py` | Answers questions like *"average marks by gender"* and shows which tool it picked |
| 7. Validate | `result_validator.py` | Warns about small data, weak models, suspicious correlations |
| 8. Report | `pdf_report_generator.py`, `report_generator.py` | Downloadable PDF and Markdown report |

`pipeline.py` runs all steps once; `app.py` is the Streamlit interface.

### How the agent decides
* **EDA** runs whenever there is data to summarise.
* **Machine Learning** runs only with **30+ rows**, a numeric target and at least one feature. The target is guessed from its name (marks, score, price, salary...) or you can choose it in the sidebar.
* **Anomaly Detection** runs with **8+ rows** and at least one numeric column.
* **ID columns** (e.g. `Student_ID`) are detected and excluded from statistics, ML and anomaly detection.

## Loading many files / folders with sub-folders

Three ways, all in the sidebar:

1. **Select many CSV files at once** in the upload dialog (open your folder, press `Ctrl+A`).
2. **Upload a ZIP** of the whole folder - every CSV inside, including sub-folders, is found.
3. **Type a folder path** (e.g. `G:\My Drive\data`) - every CSV in it and all sub-folders is loaded
   (works when the app runs on your own computer).

When more than one file is loaded you choose how the agent uses them:

* **Combine all files into one dataset** - stacked, with a `Source_File` column so you can ask
  *"average revenue by source file"*.
* **Combine files that have the same columns** - ideal when a folder mixes different kinds of files.
  This is the default when the files do not all share the same columns.
* **Analyse one file** - pick a single file.

Empty, hidden, corrupt or non-CSV files are skipped with a warning; comma, `;` and tab separated files
and Excel-style (Latin-1) encodings are read automatically.

## Setup (Windows / PowerShell)

```powershell
cd "path\to\Autonomous_Data_Analysis_Agent"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If `python` is not recognised, call it by its full path once to create the venv, e.g.
`& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" -m venv venv`.
If PowerShell blocks the activate script: `Set-ExecutionPolicy -Scope Process Bypass`.
If a package fails to install on a very new Python, create the venv with Python 3.12 instead.

## Project structure

```
Autonomous_Data_Analysis_Agent/
├── app.py                  # Streamlit interface
├── requirements.txt
├── data/
│   ├── students.csv        # your own dataset (used first if present)
│   └── sample_students_large.csv   # 200-row demo dataset
├── modules/
│   ├── utils.py            # ID / numeric / categorical column detection
│   ├── file_loader.py      # many files / ZIP / folder loading
│   ├── data_cleaner.py
│   ├── data_profiler.py
│   ├── agent.py
│   ├── agent_executor.py
│   ├── llm_service.py
│   ├── question_answering.py
│   ├── result_validator.py
│   ├── report_generator.py
│   ├── pdf_report_generator.py
│   └── pipeline.py
└── tests/
    ├── test_modules.py     # python -m pytest -q
    └── smoke_app.py        # runs app.py without a browser
```

## Questions you can ask

* What is the average marks? · Who has the highest attendance?
* Average marks by gender · How many female students?
* What affects marks? · Correlation between study hours and marks
* Are there any anomalies? · Any missing values?
* Which features matter most for the prediction?

## Run the tests

```powershell
pip install pytest
python -m pytest -q
```

## Limitations (good to mention in your report)

* The "AI" is a **transparent rule-based agent** (decisions and insights follow explicit rules); the machine-learning parts use scikit-learn.
* Machine learning is **regression only** (numeric target). Classification is a natural extension.
* Question answering understands common question patterns, not free-form language.
* Correlation is not causation; tiny datasets give unreliable statistics (the validator warns about this).

## Ideas for extending it

1. Plug a real LLM into `generate_insights` / `answer_question` (build prompts from the analysis results).
2. Add classification when the target is categorical.
3. Add time-series detection for date columns.
4. Deploy free on Streamlit Community Cloud (push to GitHub, select `app.py`).
