# StudentsPerformance/main.py

from pydantic import BaseModel
from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import uvicorn
import joblib


BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / 'model_StudentsPerformance.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_StudentsPerformance.pkl')

student_app = FastAPI()

FEATURE_ORDER = [
    'math score', 'reading score', 'avarage_score', '3 subjects',
    'gender_male',
    'race/ethnicity_group B', 'race/ethnicity_group C',
    'race/ethnicity_group D', 'race/ethnicity_group E',
    "parental level of education_bachelor's degree",
    'parental level of education_high school',
    "parental level of education_master's degree",
    'parental level of education_some college',
    'parental level of education_some high school',
    'lunch_standard',
    'test preparation course_none',
]


class Student(BaseModel):
    gender: str
    race_ethnicity: str
    parent: str
    lunch: str
    test: str
    math_score: float
    reading_score: float


@student_app.post('/predict/')
async def check_score(student: Student):
    math_score = student.math_score
    reading_score = student.reading_score
    avarage_score = round((math_score + reading_score) / 2, 2)
    three_subjects = math_score + reading_score

    row = {
        'math score': math_score,
        'reading score': reading_score,
        'avarage_score': avarage_score,
        '3 subjects': three_subjects,
        'gender_male': 1 if student.gender == 'male' else 0,
        'race/ethnicity_group B': 1 if student.race_ethnicity == 'group B' else 0,
        'race/ethnicity_group C': 1 if student.race_ethnicity == 'group C' else 0,
        'race/ethnicity_group D': 1 if student.race_ethnicity == 'group D' else 0,
        'race/ethnicity_group E': 1 if student.race_ethnicity == 'group E' else 0,
        "parental level of education_bachelor's degree": 1 if student.parent == "bachelor's degree" else 0,
        'parental level of education_high school': 1 if student.parent == 'high school' else 0,
        "parental level of education_master's degree": 1 if student.parent == "master's degree" else 0,
        'parental level of education_some college': 1 if student.parent == 'some college' else 0,
        'parental level of education_some high school': 1 if student.parent == 'some high school' else 0,
        'lunch_standard': 1 if student.lunch == 'standard' else 0,
        'test preparation course_none': 1 if student.test == 'none' else 0,
    }

    features_df = pd.DataFrame([row])[FEATURE_ORDER]

    scaled = scaler.transform(features_df)
    predict = model.predict(scaled)[0]

    return {'Прогнозируемый бал по writing score': round(predict, 2)}


if __name__ == '__main__':
    uvicorn.run(student_app, host='127.0.0.1', port=8000)
