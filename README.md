# Codveda Technologies — Data Analytics Internship

Solutions to the Codveda Technologies Data Analytics internship tasks (Levels 1–3).

### 📁 Project Structure
- `Data/` — all datasets used by the scripts
- `Level 1/`, `Level 2/`, `Level 3/` — task scripts; each script saves its charts next to itself

### ▶️ How to Run
```bash
pip install -r requirements.txt
python "Level 1/Level 1 (Task1)/Level 1 (Task1).py"
```
The Streamlit dashboard (Level 3, Task 2) is started with:
```bash
streamlit run "Level 3/Level 3 (Task 2).py"
```
Scripts locate the `Data/` folder relative to their own location, so they can be run from any directory.

---

### 🛠️ Tech Stack
- **Language:** Python
- **Data Analysis & ML:** `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `textblob`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **Dashboarding:** `streamlit`, `plotly`

---

### 🚀 Overview of Tasks

#### **Level 1**
- **Task 1: Data Cleaning & Preprocessing** — Missing value imputation, duplicate handling, cleaned CSV export.
- **Task 2: Exploratory Data Analysis (EDA)** — Summary statistics, correlations, histogram, boxplot & heatmap exports.
- **Task 3: Data Visualization** — Interactive & static bar charts, line plots, scatter plots.

#### **Level 2**
- **Task 1: Regression Analysis** — Predictive regression model for real estate dataset ($R^2$, MSE metrics).
- **Task 2: Time Series Analysis** — AAPL stock closing prices: Trend extraction, moving average calculation, time-series decomposition.
- **Task 3: Clustering Analysis** — K-Means clustering algorithm using StandardScaler and Elbow plot method.

#### **Level 3**
- **Task 1: Classification Analysis** — Logistic regression predictive model evaluation (Precision, Recall, F1).
- **Task 2: Streamlit Dashboard** — Interactive data dashboard for dataset analysis.
- **Task 3: Sentiment Analysis (NLP)** — Natural Language Processing with TextBlob, sentiment classification, WordCloud chart.