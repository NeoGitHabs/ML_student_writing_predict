# StudentsPerformance/main.py

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import joblib
import uvicorn

BASE_DIR = Path(__file__).parent

RACE_GROUPS = ["A", "B", "C", "D", "E"]
EDUCATION_LEVELS = [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree",
]


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model  = joblib.load(BASE_DIR / "model_lin_StudentsPerformance.pkl")
    app.state.scaler = joblib.load(BASE_DIR / "scaler_StudentsPerformance.pkl")
    yield


app = FastAPI(title="Writing Score Predictor", lifespan=lifespan)


# ── Schema ─────────────────────────────────────────────────────────────────────
class Student(BaseModel):
    gender:         str
    race_ethnicity: str          # "group A".."group E"
    parent:         str          # уровень образования родителя
    lunch:          str
    test:           str          # "none" | "completed"
    math_score:     float
    reading_score:  float

    @field_validator("race_ethnicity")
    @classmethod
    def validate_race(cls, v: str) -> str:
        letter = v.strip().upper().replace("GROUP ", "")
        if letter not in RACE_GROUPS:
            raise ValueError(f"race_ethnicity должен быть одним из group {RACE_GROUPS}")
        return f"group {letter}"

    @field_validator("parent")
    @classmethod
    def validate_parent(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in EDUCATION_LEVELS:
            raise ValueError(f"parent должен быть одним из: {EDUCATION_LEVELS}")
        return v


# ── Utils ──────────────────────────────────────────────────────────────────────
def build_features(s: Student) -> list[float]:
    # Порядок строго повторяет features.columns.tolist() из ноутбука —
    # без avarage_score и 3 subjects, они были удалены перед обучением.
    return [
        s.math_score,
        s.reading_score,
        1.0 if s.gender.strip().lower() == "male" else 0.0,
        1.0 if s.race_ethnicity == "group B" else 0.0,
        1.0 if s.race_ethnicity == "group C" else 0.0,
        1.0 if s.race_ethnicity == "group D" else 0.0,
        1.0 if s.race_ethnicity == "group E" else 0.0,
        1.0 if s.parent == "bachelor's degree" else 0.0,
        1.0 if s.parent == "high school" else 0.0,
        1.0 if s.parent == "master's degree" else 0.0,
        1.0 if s.parent == "some college" else 0.0,
        1.0 if s.parent == "some high school" else 0.0,
        1.0 if s.lunch.strip().lower() == "standard" else 0.0,
        1.0 if s.test.strip().lower() == "none" else 0.0,
    ]


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(student: Student):
    features = build_features(student)
    scaled   = app.state.scaler.transform([features])
    score    = float(app.state.model.predict(scaled)[0])

    return {
        "predicted_writing_score": round(score, 2),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)