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



