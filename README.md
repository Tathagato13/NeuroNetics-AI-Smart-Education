# NeuroNetics-AI-Smart-Education
NeuroNetics-SmartEdu is an AI-powered education platform that provides personalized learning, academic risk prediction, and real-time AI doubt solving using machine learning and NLP. Built with React and FastAPI.

# NeuroNetics-SmartEdu

An Intelligent AI-Based Smart Education System designed to provide personalized learning, predictive academic analytics, and real-time AI academic support. This platform helps students improve performance through adaptive study plans while enabling teachers to identify at-risk learners early using data-driven insights.

---

## 🚀 Features

- Personalized Learning Engine  
- Smart Concept-Level Weakness Detection  
- Predictive Academic Performance Analytics  
- NLP-Based AI Doubt Solving Assistant  
- Student Dashboard (Progress, Badges, Weak Topics)  
- Teacher Analytics Dashboard (Risk Alerts, Class Performance)  
- Gamification for learner engagement  

---

## 🧠 System Architecture

- React.js Frontend (Student & Teacher UI)  
- FastAPI Backend (API Management & Logic)  
- Scikit-learn (Performance Prediction & Risk Classification)  
- OpenAI / HuggingFace (AI Tutor)  
- SQLite / CSV (Data Storage)  

---

## 🛠 Technology Stack

### Frontend
- React.js  
- Axios  
- Chart.js / Recharts  

### Backend
- FastAPI  
- Python  

### Machine Learning
- Scikit-learn  

### NLP
- OpenAI API / HuggingFace Transformers  

### Database
- SQLite / CSV  

### Deployment
- Vercel (Frontend)  
- Render (Backend)  

---
# 🧠 NeuroNetics-SmartEdu
### AI-Powered Smart Education System

> An end-to-end MVP combining FastAPI, scikit-learn ML models, NLP tutoring, SQLite, and a React dashboard — built for modern academic intelligence.

---

## 🏗️ Architecture

```
React Frontend
     │
     ▼ HTTP REST
FastAPI Backend
     │         │
     ▼         ▼
ML Models    SQLite DB
(sklearn)  (students table)
     │
     ▼
NLP Tutor (rule-based / HuggingFace)
```

---


## 🚀 Quick Start

### Step 1 — Train ML Models
```bash
cd ml
pip install scikit-learn joblib numpy
python train_models.py
# Outputs: saved_models/regressor.pkl, classifier.pkl, scaler.pkl
```

### Step 2 — Start Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Step 3 — Open Frontend
```bash
# Open frontend/index.html directly in your browser
# OR serve it:
cd frontend
npx serve .   # then visit http://localhost:3000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict final score + risk level |
| GET | `/weakness/{id}` | Get weak topics for a student |
| GET | `/students` | List all students |
| GET | `/students/{id}` | Get single student |
| POST | `/chat` | AI tutor Q&A |
| GET | `/analytics` | Class-level analytics |

### Example: POST /predict
```json
{
  "name": "Priya Sharma",
  "quiz": 78,
  "attendance": 85,
  "study_time": 5,
  "math_score": 82,
  "science_score": 45,
  "english_score": 70,
  "history_score": 65
}
```

**Response:**
```json
{
  "student_id": 4,
  "predicted_final": 77.3,
  "risk": "Safe",
  "risk_probability": 18.2,
  "weak_topics": ["Science"],
  "recommendations": ["Focus on Science — score below 60%"]
}
```

---

## 🤖 ML Models

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Regressor | GradientBoostingRegressor | Predict final exam score |
| Classifier | RandomForestClassifier (200 trees) | Detect at-risk students |
| Scaler | StandardScaler | Normalize input features |

**Features used:** `quiz_score`, `attendance_percent`, `daily_study_hours`

**Model performance:**
- Regressor RMSE: ~5.7 (out of 100)
- Classifier Accuracy: ~93%

---

## 🗃️ Database Schema

```sql
CREATE TABLE students (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    quiz          REAL    NOT NULL,
    attendance    REAL    NOT NULL,
    study_time    REAL    NOT NULL,
    math_score    REAL    DEFAULT 70,
    science_score REAL    DEFAULT 70,
    english_score REAL    DEFAULT 70,
    history_score REAL    DEFAULT 70,
    final         REAL,
    risk          TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🎨 UI Features

### Student Dashboard
- Input quiz, attendance, study hours + subject scores
- View predicted final score (animated gauge)
- See at-risk probability
- Get personalized study recommendations
- See which subjects need attention (red = below 60%)

### Teacher Dashboard
- Stats overview: total, at-risk, avg scores
- Full student roster table
- Click any student → detailed profile + gauges
- Subject breakdown bar chart per student

### AI Tutor Chat (NeuroBuddy)
- Rule-based NLP tutor (works fully offline)
- Topic-aware: Math, Science, English, History
- Covers: study tips, stress, exam strategies
- Quick-suggestion buttons

---

## 🔧 Upgrade: Add HuggingFace NLP

Replace the `simple_nlp_response()` function in `backend/main.py`:

```python
from transformers import pipeline

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

def simple_nlp_response(message: str, name: str) -> str:
    prompt = f"You are a helpful academic tutor. Student {name} asks: {message}. Give a concise, helpful response."
    result = qa_pipeline(prompt, max_length=150)
    return result[0]["generated_text"]
```

Add to requirements.txt:
```
transformers==4.40.0
torch==2.2.0
```

---

## 📦 requirements.txt
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
scikit-learn==1.4.2
joblib==1.4.2
numpy==1.26.4
pydantic==2.7.1
python-multipart==0.0.9
```

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 (vanilla, no bundler) |
| Backend | FastAPI + Uvicorn |
| ML | scikit-learn (GBM + RandomForest) |
| NLP | Rule-based tutor / HuggingFace ready |
| Database | SQLite (via sqlite3) |
| Serialization | joblib |

---

*Built with ❤️ for the future of intelligent education.*


## 📂 Project Structure

NeuroNetics-SmartEdu/
│
├── backend/
│   ├── main.py                 # FastAPI application (all API endpoints)
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── index.html              # React single-file entry point
│   └── src/
│       ├── pages/
│       │   ├── StudentDashboard.jsx   # Student dashboard UI
│       │   └── TeacherDashboard.jsx   # Teacher analytics UI
│       │
│       └── components/
│           └── Chatbot.jsx            # AI doubt-solving chatbot
│
├── ml/
│   ├── train_models.py         # ML training and model saving script
│   └── saved_models/
│       ├── regressor.pkl       # Gradient Boosting final score predictor
│       ├── classifier.pkl      # Random Forest at-risk classifier
│       └── scaler.pkl          # StandardScaler
│
└── README.md                   # Project documentation



