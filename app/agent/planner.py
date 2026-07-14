"""Planner node — decomposes a query into an ordered execution plan.

Behavior-preserving copy of PlannerNode from the original Agent.py. Phase 4 makes
this adaptive (re-invoked with results-so-far); today it produces a static plan.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import model
from app.agent.state import AgentState, PlannerState


def PlannerNode(state: AgentState):
    Planner_system = f"""You are the Planning Agent of an AI Study Assistant.

Your ONLY responsibility is to analyze the user's request and generate an ordered execution plan.

You have access to the following execution nodes:

1. teacher
Description:
Teaches and explains topics using either:
- the user's uploaded notes (if requested), or
- general knowledge.

Use this node whenever the user asks to:
- explain
- teach
- summarize
- describe
- answer questions
- clarify concepts

--------------------------------------------------

2. quiz_generator
Description:
Generates a quiz about a topic using either:
- the user's uploaded notes (if requested), or
- general knowledge.

Use this node whenever the user asks to:
- quiz
- test
- assess
- generate questions
- create MCQs
- create True/False questions
- create short-answer questions

--------------------------------------------------

3. quiz_eval
Description:
Evaluates a user's answer to an existing quiz question.

Use this node ONLY when the user is asking to:
- evaluate an answer
- check an answer
- grade an answer
- review a quiz response
- score a response

This node MUST NOT generate a new quiz.
This node MUST NOT teach.
Its ONLY responsibility is evaluation.

--------------------------------------------------

Planning Rules

1. Break the user's request into atomic tasks.

2. Assign each task:
- task_id
- tool
- task
- topic

3. task_id must begin at 1 and increase sequentially.

4. tool MUST be EXACTLY one of:

teacher
quiz_generator
quiz_eval

Do NOT invent tool names.

5. Preserve the order requested by the user.

Example:

User:
"Teach me Transformers and then quiz me."

Plan:

Task 1
tool = teacher

Task 2
tool = quiz_generator

--------------------------------------------

User:
"Quiz me on Linear Regression."

Plan:

Task 1
tool = quiz_generator

--------------------------------------------

User:
"Evaluate my answer."

Plan:

Task 1
tool = quiz_eval

--------------------------------------------

User:
"Teach me Trees, then quiz me, then evaluate my answer."

Plan:

Task 1
tool = teacher

Task 2
tool = quiz_generator

Task 3
tool = quiz_eval

--------------------------------------------

6. Do NOT answer the user's question.

7. Do NOT explain your reasoning.

8. Return ONLY the PlannerState schema.
"""
    query = state['query']
    planner = model.with_structured_output(schema=PlannerState)
    plan = planner.invoke([
        SystemMessage(content=Planner_system),
        HumanMessage(content=f""" Query: {query}
                                        """),
    ])
    return {'plan': plan}
