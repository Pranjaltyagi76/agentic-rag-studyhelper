"""LLM-as-judge scorers for RAG evaluation (Phase 8.5).

Each returns an integer 1-5 (5 = best). They reuse the app's robust structured-output
helper so a flaky judge call degrades to a low score instead of crashing the eval.
"""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.structured import structured_invoke


class Judgement(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = poor, 5 = excellent")
    reason: str = Field(description="One sentence justification.")


class Abstention(BaseModel):
    abstained: bool = Field(
        description="True if the answer admits the information is not available / does not "
        "answer it; False if it asserts a specific answer."
    )
    reason: str = Field(description="One sentence justification.")


def _judge(system: str, human: str) -> int:
    result = structured_invoke(
        Judgement,
        [SystemMessage(content=system), HumanMessage(content=human)],
        default=Judgement(score=1, reason="judge call failed"),
    )
    return int(result.score)


def retrieval_relevance(question: str, reference: str, notes: str) -> int:
    """Do the retrieved notes contain the information needed to answer the question?"""
    return _judge(
        "Rate 1-5 whether the RETRIEVED NOTES contain the information needed to answer the "
        "QUESTION, using the REFERENCE as the gold standard. 5 = all needed info present and "
        "on-topic; 1 = irrelevant or missing. Return only the Judgement schema.",
        f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nRETRIEVED NOTES:\n{notes or '(none)'}",
    )


def faithfulness(answer: str, notes: str) -> int:
    """Is the answer supported by the retrieved notes (not hallucinated)?"""
    return _judge(
        "Rate 1-5 how faithful the ANSWER is to the NOTES: is every substantive claim "
        "supported by the notes? 5 = fully grounded; 1 = mostly unsupported/hallucinated. "
        "Ignore well-known common knowledge. Return only the Judgement schema.",
        f"NOTES:\n{notes or '(none)'}\n\nANSWER:\n{answer}",
    )


def abstained(question: str, answer: str) -> bool:
    """Hallucination probe: on an unanswerable question, did the system refuse to invent?

    True  = said it doesn't know / not in the notes  (GOOD)
    False = asserted a specific answer anyway        (HALLUCINATION)
    """
    result = structured_invoke(
        Abstention,
        [
            SystemMessage(content=(
                "The QUESTION cannot be answered from the user's notes. Decide whether the "
                "ANSWER correctly abstained. abstained=true if it states the information is "
                "not in the notes / it cannot answer confidently / it asks for the notes. "
                "abstained=false if it asserts a specific factual answer to the question "
                "(even hedged). Return only the Abstention schema."
            )),
            HumanMessage(content=f"QUESTION:\n{question}\n\nANSWER:\n{answer}"),
        ],
        default=Abstention(abstained=False, reason="judge call failed; counted as hallucination"),
    )
    return bool(result.abstained)


def answer_correctness(question: str, reference: str, answer: str) -> int:
    """How well does the answer match the reference in correctness and completeness?"""
    return _judge(
        "Rate 1-5 how well the ANSWER matches the REFERENCE for the QUESTION in correctness "
        "and completeness. 5 = fully correct and complete; 1 = incorrect or unrelated. "
        "Return only the Judgement schema.",
        f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}",
    )
