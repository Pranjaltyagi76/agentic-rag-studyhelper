"""Quiz nodes — generation and evaluation.

Behavior-preserving copy from the original Agent.py. Phase 3 shares the retrieval
subgraph with the teacher node.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import model
from app.agent.retrieval import run_retrieval
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


def quiz_generator_node(state: AgentState):
    """Generate a quiz from the user's notes (self-correcting retrieval) or general knowledge.

    Phase 3: the notes-retrieval decision + query now come from the shared
    self-correcting retrieval subgraph (grade + rewrite/retry). Quiz stays grounded in
    notes, so web fallback is disabled (``allow_web=False``); if notes are insufficient
    the quiz system prompt falls back to general knowledge.
    """
    ctx = run_retrieval(
        session_id=state["session_id"],
        query=state["query"],
        uploaded_files=state["uploaded_files"],
        purpose="quiz",
        allow_web=False,
    )

    quiz = QuizGenerator.invoke([
        SystemMessage(content=quiz_system),
        HumanMessage(content=f"""
User Query:

{state["query"]}

Retrieved Notes:

{ctx.get("notes_text", "")}
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
