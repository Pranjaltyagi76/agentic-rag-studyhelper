"""Shared LLM instance (Groq).

Single source of the reasoning model so every node uses the same configured client
(strategy.md: "one place per concern").
"""

from langchain_groq import ChatGroq

from app.config import settings

model = ChatGroq(model=settings.GROQ_MODEL)
