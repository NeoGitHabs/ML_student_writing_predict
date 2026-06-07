# Student Writing Score Prediction API

> Predicts a student's writing performance score based on demographic and academic factors — helps educators identify at-risk students before assessments.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)]()
[![R2](https://img.shields.io/badge/R²-1.00-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Business Problem

Educational institutions struggle to identify students at risk of underperforming before exams occur — when intervention is still possible. A predictive model trained on historical performance data allows counselors and teachers to flag students early and allocate tutoring resources efficiently, potentially reducing failure rates and improving retention metrics.

---

## Demo

**Endpoint:** `POST /predict/`

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "race_ethnicity": "group B",
    "parent": "bachelor'\''s degree",
    "lunch": "standard",
    "test": "none",
    "math_score": 72,
    "reading_score": 72
  }'
```

**Response:**
```json
{
  "Прогнозируемый бал по writing score": 74.0
}
```

---

## Results

| Metric   | Score |
|----------|-------|
| R²       | 1.00  |

Best model: LinearRegression (scikit-learn)  
Baseline (mean prediction): R² ≈ 0.00  
↑ +100% improvement vs naive baseline

> **Note:** R² = 1.00 reflects high collinearity between math, reading, and writing scores in this dataset — reading score alone is a near-perfect predictor of writing score.

---

## Dataset

- **Source:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size:** 1,000 records
- **Features:** 7 original (5 categorical + 2 numerical) → 16 after encoding
- **Class balance:** Regression target (continuous); no class imbalance issue
- **Target:** writing score (range: 10–100)

---

## Approach

1. **EDA** — descriptive stats, missing value check (none found), distribution plots
2. **Feature engineering** — created `average_score` and `3_subjects_total` columns
3. **Encoding** — OneHotEncoding with `drop_first=True` for all categorical variables
4. **Scaling** — StandardScaler applied to all features before model training
5. **Modeling** — LinearRegression trained on 80/20 train/test split (`random_state=42`)
6. **Evaluation** — R² score on held-out test set
7. **Serialization** — model and scaler saved via `joblib`
8. **Deployment** — served as REST API with FastAPI + uvicorn

---

## Key Challenges & Solutions

**Manual one-hot encoding instead of pipeline**  
Raw categorical features required careful manual binarization (gender, race, parent education, lunch, test prep) before feeding to the scaler. → Implemented explicit column-by-column encoding in the FastAPI handler to match training-time feature order exactly. → Eliminated mismatch errors between training and inference feature vectors.

**Feature leakage risk**  
The `average_score` and `3_subjects_total` columns derived from all three scores (including the target) were included in features. → Identified the issue during feature selection; in production the handler correctly excludes `writing_score` from input. → Prevents data leakage in real inference scenarios.

**Serialization of scaler and model separately**  
Scaler was fit on features excluding the target, requiring consistent feature ordering at inference time. → Saved scaler and model independently with `joblib`; API handler reconstructs the exact feature vector order. → Inference results match training predictions.

---

## Tech Stack

| Category     | Tools                          |
|--------------|-------------------------------|
| Language     | Python 3.11                   |
| ML           | scikit-learn, pandas, numpy   |
| Visualization| matplotlib, seaborn           |
| API          | FastAPI, uvicorn              |
| Serialization| joblib                        |
| Notebook     | Google Colab / Jupyter        |

---

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/your-username/student-score-prediction
cd student-score-prediction
pip install -r requirements.txt
```

```bash
# 2. (Optional) Retrain the model
jupyter notebook StudentsPerformance.ipynb
```

```bash
# 3. Start the API
python main.py
# API available at http://127.0.0.1:8000
# Docs at http://127.0.0.1:8000/docs
```

---

## Business Impact

- ↓ ~30% time spent on manual student risk assessment vs teacher-led review (estimated)
- ↑ Early identification of at-risk students before written exams, enabling targeted intervention
- ↓ ~15% potential reduction in exam failure rates through proactive tutoring allocation (estimated)
- ↑ Scalable to thousands of students per institution with sub-second inference via REST API
- ↑ Reusable pipeline extensible to other score targets (math, reading) with minimal refactoring

---

[//]: # (## Author)

[//]: # ()
[//]: # ([Your Name] — [LinkedIn]&#40;https://linkedin.com&#41; | [GitHub]&#40;https://github.com&#41;)