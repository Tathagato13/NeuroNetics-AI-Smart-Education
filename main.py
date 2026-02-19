"""
NeuroNetics-SmartEdu – FastAPI Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3, joblib, numpy as np, os, pathlib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent
MODEL_DIR = BASE.parent / "ml" / "saved_models"
DB_PATH   = BASE / "neuronetics.db"

# ── Load ML models ───────────────────────────────────────────────────────────
reg    = joblib.load(MODEL_DIR / "regressor.pkl")
clf    = joblib.load(MODEL_DIR / "classifier.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# ── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            quiz        REAL    NOT NULL,
            attendance  REAL    NOT NULL,
            study_time  REAL    NOT NULL,
            math_score  REAL    DEFAULT 70,
            science_score REAL  DEFAULT 70,
            english_score REAL  DEFAULT 70,
            history_score REAL  DEFAULT 70,
            final       REAL,
            risk        TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Schemas ───────────────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    name:          str
    quiz:          float
    attendance:    float
    study_time:    float
    math_score:    Optional[float] = 70.0
    science_score: Optional[float] = 70.0
    english_score: Optional[float] = 70.0
    history_score: Optional[float] = 70.0

class ChatInput(BaseModel):
    message: str
    student_name: Optional[str] = "Student"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="NeuroNetics-SmartEdu API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper ────────────────────────────────────────────────────────────────────
def predict_student(quiz, attendance, study_time):
    X = scaler.transform([[quiz, attendance, study_time]])
    final  = float(np.clip(reg.predict(X)[0], 0, 100))
    risk_i = int(clf.predict(X)[0])
    risk   = "At-Risk" if risk_i == 1 else "Safe"
    proba  = float(clf.predict_proba(X)[0][1])
    return final, risk, proba

def detect_weakness(math, science, english, history):
    topics = {"Math": math, "Science": science, "English": english, "History": history}
    return [t for t, s in topics.items() if s < 60]

def simple_nlp_response(message: str, name: str) -> str:
    """Rule-based NLP tutor (no external API needed for offline MVP)."""
    msg = message.lower()
    responses = {
        ("study", "how"): f"Hi {name}! Try the Pomodoro technique: 25 min focused study → 5 min break. Repeat 4x then take a longer break.",
        ("math",): f"For math, {name}, practice daily problem sets. Khan Academy and Wolfram Alpha are great resources!",
        ("science",): f"Science is best learned through curiosity! Try YouTube channels like CrashCourse or Kurzgesagt, {name}.",
        ("english", "writing"): f"Improve writing by reading widely and writing daily, {name}. The Hemingway App helps simplify prose.",
        ("history",): f"History becomes vivid when you connect events. Try timelines and documentaries, {name}!",
        ("stress", "anxious", "worried"): f"It's okay to feel stressed, {name}. Take a breath, break tasks into smaller steps, and remember to sleep well.",
        ("exam", "test", "quiz"): f"For exams, {name}: review notes within 24h, do practice tests, and sleep 8h before the test.",
        ("weak", "failing", "bad"): f"Don't give up, {name}! Identify which topics confuse you most and ask your teacher or tutor for help on those specifically.",
        ("recommend", "suggest"): f"Based on your profile, {name}, I'd recommend focusing on your weakest topics first, then reviewing stronger ones to maintain them.",
        ("hello", "hi", "hey"): f"Hello {name}! I'm your AI tutor. Ask me about study tips, subjects, or exam strategies!",
    }
    for keywords, reply in responses.items():
        if any(k in msg for k in keywords):
            return reply
    return (
        f"Great question, {name}! While I'm a simple rule-based tutor, I suggest: "
        "1) Break the topic into small parts, 2) Find examples online, "
        "3) Teach it back to yourself out loud. That's the Feynman technique!"
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict_endpoint(data: StudentInput):
    final, risk, proba = predict_student(data.quiz, data.attendance, data.study_time)
    weak = detect_weakness(data.math_score, data.science_score, data.english_score, data.history_score)
    
    # Save to DB
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO students 
           (name, quiz, attendance, study_time, math_score, science_score, english_score, history_score, final, risk)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (data.name, data.quiz, data.attendance, data.study_time,
         data.math_score, data.science_score, data.english_score, data.history_score,
         round(final, 2), risk)
    )
    student_id = cur.lastrowid
    conn.commit()
    conn.close()

    # Study recommendations
    recommendations = []
    if data.attendance < 70:
        recommendations.append("Attend more classes — attendance < 70% strongly impacts final grades.")
    if data.study_time < 4:
        recommendations.append("Increase daily study time to at least 4 hours.")
    if data.quiz < 60:
        recommendations.append("Revise quiz material; low quiz scores signal concept gaps.")
    for topic in weak:
        recommendations.append(f"Focus on {topic} — score below 60%, needs immediate attention.")
    if not recommendations:
        recommendations.append("Great performance! Keep maintaining your current study habits.")

    return {
        "student_id":     student_id,
        "predicted_final": round(final, 2),
        "risk":           risk,
        "risk_probability": round(proba * 100, 1),
        "weak_topics":    weak,
        "recommendations": recommendations,
    }

@app.get("/weakness/{student_id}")
def weakness_endpoint(student_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")
    weak = detect_weakness(row["math_score"], row["science_score"], row["english_score"], row["history_score"])
    topic_scores = {
        "Math": row["math_score"],
        "Science": row["science_score"],
        "English": row["english_score"],
        "History": row["history_score"],
    }
    return {"student_id": student_id, "weak_topics": weak, "topic_scores": topic_scores}

@app.get("/students")
def students_endpoint():
    conn = get_db()
    rows = conn.execute("SELECT * FROM students ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/students/{student_id}")
def get_student(student_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")
    return dict(row)

@app.post("/chat")
def chat_endpoint(body: ChatInput):
    reply = simple_nlp_response(body.message, body.student_name)
    return {"reply": reply}

@app.get("/analytics")
def analytics_endpoint():
    conn = get_db()
    rows = conn.execute("SELECT * FROM students").fetchall()
    conn.close()
    if not rows:
        return {"total": 0, "at_risk": 0, "safe": 0, "avg_final": 0, "avg_quiz": 0, "avg_attendance": 0}
    data = [dict(r) for r in rows]
    total = len(data)
    at_risk = sum(1 for r in data if r["risk"] == "At-Risk")
    return {
        "total":          total,
        "at_risk":        at_risk,
        "safe":           total - at_risk,
        "avg_final":      round(sum(r["final"] or 0 for r in data) / total, 1),
        "avg_quiz":       round(sum(r["quiz"] for r in data) / total, 1),
        "avg_attendance": round(sum(r["attendance"] for r in data) / total, 1),
        "students":       data,
    }

@app.get("/")
def root():
    return {"message": "NeuroNetics-SmartEdu API is running 🧠"}