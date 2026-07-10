# 📚 Study AI Agent

> 🚀 An Agentic AI-powered Study Assistant built with **LangGraph**, **FastAPI**, **ChromaDB**, and **LLMs** that can teach concepts, generate quizzes, evaluate answers, and learn from uploaded notes using Retrieval-Augmented Generation (RAG).

---

## ✨ Features

🎓 **AI Teacher**
- Explains concepts in an easy-to-understand manner
- Can teach using its own knowledge or your uploaded notes
- Supports Markdown formatted lessons

🧠 **Agentic Planning**
- Uses an LLM Planner to decompose user requests into executable tasks
- Deterministic workflow execution using LangGraph

📄 **Retrieval-Augmented Generation (RAG)**
- Upload PDFs or study notes
- Stores embeddings in ChromaDB
- Retrieves only relevant information for teaching and quiz generation

📝 **Quiz Generation**
- Generates:
  - ✅ MCQs
  - ✅ True/False
  - ✅ Short Answer Questions

📊 **Quiz Evaluation**
- MCQs & True/False
  - Correct / Wrong
  - Detailed explanation
- Short Answer
  - Rating (0–5 ⭐)
  - Personalized feedback

🖥️ **Interactive Frontend**
- Teacher Center
- Quiz Center
- Question Navigation
- Answer Submission
- Real-time Evaluation

---

# 🏗️ Architecture

```text
                User
                  │
                  ▼
          🧠 Planner Node
                  │
                  ▼
          Deterministic Router
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
   📖 Teacher  📝 Quiz   ✅ Evaluator
        │         │         │
        └─────────┴─────────┘
                  │
                  ▼
            🌐 Frontend UI
```

---

# ⚙️ Tech Stack

- 🐍 Python
- ⚡ FastAPI
- 🧠 LangGraph
- 🔗 LangChain
- 🤖 Google Gemini / Groq
- 📄 ChromaDB
- 🔍 Retrieval-Augmented Generation (RAG)
- 🎨 HTML
- 💻 JavaScript
- 📝 Marked.js

---

# 🚀 How it Works

1️⃣ User sends a request.

2️⃣ The **Planner Node** analyzes the request and creates a structured execution plan.

3️⃣ The deterministic workflow routes execution to the appropriate node.

Possible nodes include:

- 📖 Teacher
- 📝 Quiz Generator
- ✅ Quiz Evaluator

4️⃣ If uploaded notes are required, the Teacher or Quiz Generator automatically performs RAG retrieval.

5️⃣ The frontend displays:
- AI lesson
- Interactive quiz
- Evaluation and feedback

---

# 📸 Screenshots

> *(Add screenshots here)*

### 📖 Teacher Center

![Teacher](screenshots/teacher.png)

### 📝 Quiz Generation

![Quiz](screenshots/quiz.png)

### ✅ Quiz Evaluation

![Evaluation](screenshots/evaluation.png)

### 📄 Learning from Uploaded Notes

![RAG](screenshots/rag.png)

---

# 📂 Project Structure

```text
Study-AI-Agent
│
├── app.py
├── Final_Agent_Deterministic.py
├── ingest.py
├── database.py
├── StudySpace_FINAL.html
├── uploads/
├── chromadb/
├── requirements.txt
└── README.md
```

---

# 🛠️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Study-AI-Agent.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env`

```env
GOOGLE_API_KEY=YOUR_API_KEY
GROQ_API_KEY=YOUR_API_KEY
```

Run FastAPI

```bash
uvicorn app:app --reload
```

Open

```
StudySpace_FINAL.html
```

---

# 🎯 Future Improvements

- 📚 Flashcard Generation
- 🧠 Long-term Memory
- 🎙️ Voice Interaction
- 📈 Learning Analytics Dashboard
- 📅 Study Planner
- 🎥 YouTube Lecture Retrieval
- 📖 Multi-document Reasoning

---

# 🤝 Contributions

Contributions, ideas, and suggestions are always welcome!

Feel free to open an issue or submit a pull request.

---

# ⭐ If you like this project...

Give it a ⭐ on GitHub!

It really helps and motivates further development 🚀
