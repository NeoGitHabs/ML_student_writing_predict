from pydantic import BaseModel
from fastapi import FastAPI
from pathlib import Path
import uvicorn
import joblib


BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / 'model_StudentsPerformance.pkl')
scaler = joblib.load(BASE_DIR / 'scaler_StudentsPerformance.pkl')

student_app = FastAPI()

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
    student_dict = dict(student)

    math_score = student_dict['math_score']
    reading_score = student_dict['reading_score']
    avarage_score = round((math_score + reading_score) / 2, 2)
    three_subjects = math_score + reading_score

    new_gender = student_dict.pop('gender')
    gender_binar = [1 if new_gender == 'male' else 0]

    new_race_ethnicity = student_dict.pop('race_ethnicity')
    race_ethnicity_binar = [
        1 if new_race_ethnicity == 'group B' else 0,
        1 if new_race_ethnicity == 'group C' else 0,
        1 if new_race_ethnicity == 'group D' else 0,
        1 if new_race_ethnicity == 'group E' else 0,
    ]

    new_parent = student_dict.pop('parent')
    parent_binar = [
        1 if new_parent == "bachelor's degree" else 0,
        1 if new_parent == 'high school' else 0,
        1 if new_parent == "master's degree" else 0,
        1 if new_parent == 'some college' else 0,
        1 if new_parent == 'some high school' else 0,
    ]

    new_lunch = student_dict.pop('lunch')
    lunch_binar = [1 if new_lunch == 'standard' else 0]

    new_test = student_dict.pop('test')
    test_binar = [1 if new_test == 'none' else 0]

    features = (
        [math_score, reading_score, avarage_score, three_subjects]
        + gender_binar
        + race_ethnicity_binar
        + parent_binar
        + lunch_binar
        + test_binar
    )
    scaled = scaler.transform([features])
    predict = model.predict(scaled)[0]
    return {'Прогнозируемый бал по writing score': round(predict, 2)}

if __name__ == '__main__':
    uvicorn.run(student_app, host='127.0.0.1', port=8000)
