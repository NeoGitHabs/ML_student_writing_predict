from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn


model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

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

    features = list(student_dict.values()) + gender_binar + race_ethnicity_binar + parent_binar + lunch_binar + test_binar
    scaled = scaler.transform([features])
    predict = model.predict(scaled)[0]
    return {'Прогнозируемый бал по writing score': {round(predict, 2)}} # Словарь — стандарт для REST API, потому что JSON легко парсится на любом языке.

if __name__ == '__main__':
    uvicorn.run(student_app, host='127.0.0.1', port=8000)


# для теста
# {
#   "gender": "female",
#   "race_ethnicity": "group B",
#   "parent": "bachelor's degree",
#   "lunch": "standard",
#   "test": "none",
#   "math_score": 72,
#   "reading_score": 72
# }
