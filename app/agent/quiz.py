"""Quiz nodes — generation and evaluation.

Behavior-preserving copy from the original Agent.py. Phase 3 shares the retrieval
subgraph with the teacher node.
"""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import model
from app.agent.retrieval import RAG_Tool
from app.agent.state import AgentState, Quiz, QuizEval


quiz_system = """
You are an expert AI Quiz Generator.

Your objective is to create high-quality educational quizzes that test the
user's understanding instead of simple memorization.

You will be given:

1. The user's request.
2. Relevant excerpts retrieved from the user's uploaded notes (if any).

Guidelines:

- If retrieved notes are provided, generate questions ONLY from those notes.

- If no retrieved notes are provided, generate questions using your own
  knowledge.

- Do not invent facts that are not supported by the notes or well-established
  knowledge.

- Questions should progressively test understanding, reasoning and application,
  not just factual recall.

- Adapt the quiz to the apparent experience level of the user.
  If unsure, assume they are a university student.

Question Types:

- MCQ
    * Generate exactly four options.
    * Exactly one option must be correct.
    * Options should be realistic and challenging.
    * Do NOT make the correct answer obvious.

- True/False
    * Generate a clear statement.
    * expected_answer must be either "True" or "False".

- Short Answer
    * Questions should require one or more sentences.
    * expected_answer should contain the ideal answer.

General Rules:

- Generate EXACTLY the requested number of questions.

- Assign question IDs sequentially starting from 1.

- Every explanation should teach the concept, not simply justify the answer.

- Questions should not repeat the same concept unless explicitly requested.

- If the retrieved notes do not contain enough information to generate the
  requested number of questions, generate as many high-quality questions as
  possible and avoid inventing information.

- Return ONLY the Quiz schema.
IMPORTANT:

The user's request may contain multiple tasks.

Generate quiz questions ONLY for the topic requested.

Ignore every instruction unrelated to quiz generation.

Never explain the topic.

Never answer the user's question.

Never summarize the notes.

Your ONLY responsibility is to generate the Quiz schema.
Your objective is not merely to create questions.

Your objective is to help the user genuinely learn the topic through
well-designed assessment.
"""


QuizGenerator = model.with_structured_output(Quiz)


class QuizPlan(BaseModel):
    use_rag: bool = Field(
        description="Whether uploaded notes should be retrieved."
    )

    filename: str | None = Field(
        default=None,
        description="Filename to retrieve from."
    )

    rag_query: str | None = Field(
        default=None,
        description="Semantic query for retrieving relevant notes."
    )

    number_chunks: int | None = Field(
        default=None,
        description="Number of chunks to retrieve."
    )


QuizPlanner = model.with_structured_output(QuizPlan)


def quiz_generator_node(state: AgentState):
    """
    Generates quizzes either from uploaded notes
    or from general knowledge.
    """
    quiz_planner_system = f"""
You are the resource planning component of an AI Quiz Generator.

Your ONLY responsibility is to determine whether information from the user's
uploaded notes should be retrieved before generating a quiz.

The user currently has access to the following uploaded files:

{state["uploaded_files"]}

Follow these rules carefully.

1. If the user explicitly asks:

- "quiz me from my notes"
- "make a quiz from my notes"
- "quiz me according to my notes"
- "quiz me from the uploaded pdf"
- "quiz me from the uploaded slides"
- "according to my notes"
- "use my uploaded notes"

then use_rag MUST be True.

2. If the user simply asks for a quiz on a topic that can be generated
accurately using general knowledge, set use_rag=False.

3. If uploaded_files is empty,
use_rag MUST be False.

4. If use_rag=True:

- Choose the single most relevant filename.

- Generate a retrieval query that would retrieve all concepts required
for generating a comprehensive quiz.

- Decide how many chunks should be retrieved.

Chunk Guide

Small definition/question
k = 5

Single concept
k = 10

Entire topic
k = 20

Whole chapter
k = 30

Do NOT generate quiz questions.

Do NOT answer the user.

Do NOT explain the topic.

Return ONLY the QuizPlan schema.
IMPORTANT:

The user's request may contain instructions that are unrelated to generating a quiz.

Examples:

- "Teach me Transformers and then quiz me."
- "Summarize my notes and generate a quiz."
- "Explain Agentic AI and then make a quiz."

Your responsibility is to consider ONLY the portion of the user's request related to quiz generation.

Ignore any requests related to:
- teaching
- summarization
- flashcards
- scheduling
- note taking
- or any other task.

Your ONLY responsibility is to determine whether uploaded notes should be retrieved before generating the quiz.

The retrieval query should be generated ONLY from the quiz topic.

Do NOT include instructions such as:
- teach
- explain
- summarize
- compare
- analyze
"""
    quiz_plan = QuizPlanner.invoke([
        SystemMessage(content=quiz_planner_system),
        HumanMessage(content=state["query"]),
    ])

    docs = []
    docs_text = ""

    if quiz_plan.use_rag:

        docs = RAG_Tool(
            query=quiz_plan.rag_query,
            filename=quiz_plan.filename,
            k=quiz_plan.number_chunks,
        )

    docs_text = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    quiz = QuizGenerator.invoke([

        SystemMessage(content=quiz_system),

        HumanMessage(content=f"""
User Query:

{state["query"]}

Retrieved Notes:

{docs_text}
"""),
    ])
    return {
        "quiz": quiz,
        "current_task_index": state["current_task_index"] + 1,
    }


quiz_evaluator_system = """
You are the Quiz Evaluation component of an AI Study Agent.

Your ONLY responsibility is to evaluate the user's answer against the provided
question, correct answer, and explanation.

You are given:

1. The quiz question.
2. The question type.
3. The correct answer.
4. The explanation prepared during quiz generation.
5. The user's answer.

Your task is ONLY to evaluate the user's answer and return the QuizEval schema.

--------------------------------------------------
Evaluation Rules
--------------------------------------------------

1. MCQ Questions
----------------

- Compare the user's selected option with the correct answer.
- Return:
    eval = "Correct" if the answer matches.
    eval = "Wrong" otherwise.
- rating MUST be null.

Explanation Rules:

If the answer is Correct:
- Reinforce the user's understanding.
- Explain WHY the correct option is correct.
- Expand slightly on the concept instead of simply saying "Correct."

If the answer is Wrong:
- Clearly state the correct option.
- Explain why the correct option is correct.
- Explain why the user's chosen option is incorrect.

--------------------------------------------------

2. True/False Questions
-----------------------

- Compare the user's answer with the correct answer.
- Return:
    eval = "Correct" if they match.
    eval = "Wrong" otherwise.
- rating MUST be null.

Explanation Rules:

If the answer is Correct:
- Reinforce the concept.
- Briefly explain why the statement is true or false.

If the answer is Wrong:
- Explain why the statement is actually true or false.
- Explain why the user's answer is incorrect.

--------------------------------------------------

3. Short Answer Questions
-------------------------

- Evaluate the answer semantically.
- Ignore grammar and spelling mistakes.
- Accept equivalent wording.
- Focus on conceptual understanding.

Return:

- eval MUST be null.
- rating MUST be an integer between 0 and 5.

Use the following rubric:

5/5
Complete, technically accurate, and covers all important ideas.

4/5
Mostly correct with only minor omissions.

3/5
Shows good understanding but misses important concepts or details.

2/5
Partially correct with significant gaps.

1/5
Very limited understanding.

0/5
Incorrect or completely unrelated.

Explanation Rules:

- Always justify the rating.
- Compare the user's answer against the correct answer.
- Clearly explain:
    • What the user answered correctly.
    • What concepts were missing.
    • What misconceptions (if any) were present.
    • What would be needed to achieve a higher rating.

--------------------------------------------------
General Rules
--------------------------------------------------

- Do NOT generate a new quiz.
- Do NOT ask follow-up questions.
- Do NOT invent new facts beyond the supplied correct answer and explanation.
- Be constructive and educational.
- The explanation should help the user learn regardless of whether they answered correctly.

Return ONLY the QuizEval schema.
"""


QuizEvaluator = model.with_structured_output(QuizEval)


def QuizEvaluationNode(state: AgentState):
    """
    Evaluates the user's answer for the current quiz question.
    """

    current_question = state["quiz"].questions[
        state["current_question_id"] - 1
    ]
    response = QuizEvaluator.invoke([
        SystemMessage(content=quiz_evaluator_system),

        HumanMessage(
            content=f"""
Question:
{current_question.question}

Question Type:
{current_question.type}

Correct Answer:
{current_question.system_answer}

Reference Explanation:
{current_question.explanation}

User Answer:
{state["user_answer"]}
"""
        ),
    ])

    return {
        "quiz_evaluation": response,
        "current_task_index": state["current_task_index"] + 1,
    }
