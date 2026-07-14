"""Top-level StateGraph assembly.

Behavior-preserving copy of the graph wiring from the original Agent.py. Exposes the
compiled ``agent``. Phase 4 makes ``executor`` an adaptive re-planner; Phase 5 wraps
this with a PostgresSaver checkpointer.
"""

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.planner import PlannerNode
from app.agent.teacher import teacher_node
from app.agent.quiz import quiz_generator_node, QuizEvaluationNode


def executor(state: AgentState):
    if state["current_task_index"] >= len(state["plan"].tasks):
        return END

    task = state["plan"].tasks[
        state["current_task_index"]
    ]

    return task.tool


graph = StateGraph(AgentState)

graph.add_node('planner', PlannerNode)
graph.add_node('teacher', teacher_node)
graph.add_node('quiz_generator', quiz_generator_node)
graph.add_node('quiz_eval', QuizEvaluationNode)


graph.add_edge(START, 'planner')
graph.add_conditional_edges(
    "planner",
    executor,
    {
        "teacher": "teacher",
        "quiz_generator": "quiz_generator",
        "quiz_eval": "quiz_eval",
        END: END,
    },
)

graph.add_edge("teacher", "planner")
graph.add_edge("quiz_generator", "planner")
graph.add_edge("quiz_eval", "planner")

agent = graph.compile()
