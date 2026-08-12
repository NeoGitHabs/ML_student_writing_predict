# Student Writing Score Prediction

> LinearRegression-модель для предсказания оценки за письмо (writing score) на основе демографических и академических данных 1000 студентов. R² = 0.94 на тестовой выборке.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)]()
[![R2](https://img.shields.io/badge/R²-0.94-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Problem

Образовательные учреждения не могут заранее выявить студентов с риском провала — до экзамена. Предсказание оценки по письму на основе уже известных факторов (пол, уровень образования родителей, тип питания, курс подготовки, оценки по математике и чтению) позволяет преподавателям вмешаться заблаговременно.

---

## Results

| Metric | Score |
|--------|-------|
| R²     | 0.94  |

Модель: `LinearRegression` (scikit-learn)  
Разбивка: 80/20, `random_state=42`  
Baseline (среднее): R² ≈ 0.00

> **Почему высокий R²:** Writing score сильно коррелирует с reading score и math score — это ожидаемо для данного датасета.

---

## Dataset

- **Source:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Размер:** 1 000 записей, 0 пропусков
- **Фичи:** 7 исходных → 14 после кодирования
- **Target:** `writing score` (диапазон: 10–100)

**Найденные инсайты из EDA:**
- Женщины лучше по reading/writing, мужчины по math
- Студенты с `lunch = standard` показывают оценки выше на ~10 баллов
- Прохождение курса подготовки даёт +5.6 по математике
- 3 студента из group E набрали по 300 из 300 баллов
- Образование родителей положительно коррелирует с успеваемостью

---

## Pipeline
EDA → Feature Engineering → OneHotEncoding → StandardScaler → LinearRegression → joblib

1. **EDA** — `describe()`, `isnull()`, groupby-агрегации, histplot / boxenplot / violinplot / barplot
2. **Feature engineering** — `average_score = (math + reading + writing) / 3`, `3_subjects = sum`
3. **Encoding** — `pd.get_dummies(drop_first=True)` для 5 категориальных колонок
4. **Scaling** — `StandardScaler` на 14 признаках (без target и производных)
5. **Model** — `LinearRegression`, обучение на 800 записях
6. **Save** — `model_lin_StudentsPerformance.pkl` + `scaler_StudentsPerformance.pkl`

---

## Feature List (после кодирования)
math score, reading score,
gender_male,
race/ethnicity_group B/C/D/E,
parental level of education_bachelor's/high school/master's/some college/some high school,
lunch_standard,
test preparation course_none

---

## Tech Stack

| Category      | Tools                        |
|---------------|------------------------------|
| Language      | Python 3.11                  |
| ML            | scikit-learn, pandas, numpy  |
| Visualization | matplotlib, seaborn          |
| Serialization | joblib                       |
| Notebook      | Google Colab (GPU T4)        |

---

## How to Run

```bash
git clone https://github.com/your-username/student-score-prediction
cd student-score-prediction
pip install -r requirements.txt
```

Запустить ноутбук в Google Colab:  
[Open in Colab](https://colab.research.google.com)

---

## Files
StudentsPerformance.ipynb # EDA + modeling
StudentsPerformance.csv # исходный датасет
students_clean.csv # очищенный датасет
model_lin_StudentsPerformance.pkl # обученная модель
scaler_StudentsPerformance.pkl # fitted scaler
avarage_score.png # график распределения

---

## Business Impact

- Раннее выявление студентов с риском плохой оценки по письму
- Автоматизация ручной оценки — с sub-second инференсом через joblib
- Pipeline расширяется на предсказание math/reading score без переписывания кода

---
## Struture
```
StudentsPerformance/
├── StudentsPerformance.ipynb
├── Test.txt
├── avarage_score.png
├── dataset/
│   ├── DSStudentsPerformance.docx
│   └── StudentsPerformance.csv
├── main.py
├── model_lin_StudentsPerformance.pkl
├── scaler_StudentsPerformance.pkl
└── students_clean.csv
```

---
