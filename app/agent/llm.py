"""Shared LLM instance (Groq).

Single source of the reasoning model so every node uses the same configured client
(strategy.md: "one place per concern").
"""

from langchain_groq import ChatGroq

from app.config import settings

# Reasoning model — the actual answers (teaching, quizzes, planning).
model = ChatGroq(model=settings.GROQ_MODEL)

# Cheap classifier for the grading step (model cascading). Separate Groq quota bucket,
# so relevance-grading no longer competes with generation for the daily token budget.
grader_model = ChatGroq(model=settings.GRADER_MODEL)
