from langgraph.graph import StateGraph , START, END
from typing_extensions import TypedDict , Annotated,Literal,Optional


from pydantic import BaseModel,Field


from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq


model = ChatGroq(
    model="llama-3.3-70b-versatile"
)
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage , BaseMessage
from langgraph.graph.message import add_messages


class task(BaseModel):
    task_id:int = Field(description="Give task their task id in order of their occurence")
    tool: Literal["teacher","quiz_generator","quiz_eval"] = Field(description="Node to execute this task")
    task:str = Field(description="what the task is and what tools need to be called")
    topic:str = Field(description="The topic being discussed in the query")
class PlannerState(BaseModel):
    tasks:list[task]


class QuizQuestion(BaseModel):
    id: int
    question: str
    system_answer: str
    type: Literal["mcq", "true_false", "short_answer"]
    explanation: str
    options: list[str] | None = None
class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]


class QuizEval(BaseModel):
    explanation:str
    eval:Literal['Correct','Wrong'] | None
    rating: Optional[int] = Field(default=None,
        ge=0,
        le=5,
        description="Understanding rating from 0 to 5."
    )


class AgentState(TypedDict):
    query:str
    plan:PlannerState | None
    uploaded_files : list[str]
    lesson:str | None
    quiz : Quiz | None
    current_question_id: int | None
    user_answer: str | None
    quiz_evaluation: QuizEval | None
    current_task_index: int | None
    messages: Annotated[list[BaseMessage],add_messages]


def PlannerNode(state:AgentState):
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
    plan = planner.invoke([SystemMessage(content=Planner_system),
                           HumanMessage(content=f""" Query: {query}
                                        """)])
    return {'plan':plan}


from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class TeacherPlan(BaseModel):
    use_rag : bool = Field(description="True if information is required from the user's  uploaded documents based on the query given")
    number_chunks:int | None =  Field(default = None,description="Number of relevant documents to be fetched based on the query") 
    filename:str | None = Field(default=None,description="Select one filename from the uploaded_files list that contains the most relevant information.")
    rag_query:str | None =  Field(default=None , description="The query sent to the retriever to fetch the relevant documents to the query")
    use_web: bool | None = Field(default=None,description="Do we need to research for the given query or not")
    web_queries : list[str] | None = Field(default=None,description="The exact query that we need to send to the web to get the desired content")


from database import vectordb


def RAG_Tool(query:str,vectorestore:Chroma,filename:str,k:int):
    """RAG TOOL is a tool that fetches relevant documents to the query from the uploaded documents by the user"""
    retriever = vectorestore.as_retriever(
    search_kwargs={
        "k": k,
        "filter": {
            "file_name": filename
        }
    }
)
    docs = retriever.invoke(query)
    return docs


from langchain_tavily import TavilySearch

search = TavilySearch(max_results=5)


def research_tool(queries: list[str]):
    """
    Searches the web for up-to-date information.

    Use this tool only when the user's uploaded notes are
    insufficient or outdated.

    Args:
        query: A concise web search query.

    Returns:
        Relevant web search results.
    """
    results = []
    for query in queries:
        results += search.invoke(query)
    return results


web_research_system = """
You are the research planning component of an AI Teacher.

Your job is to determine whether additional information from the web is
required before teaching the user.

You will receive:

1. The user's query.
2. The retrieved documents from the user's uploaded notes if provided.

Your responsibilities are:

1. Carefully examine the retrieved notes if given.

2. Determine whether the notes alone are sufficient to answer the user's
request accurately and completely.

3. If the notes are sufficient:
   - Set use_web=False.
   - Leave web_query empty.

4. If the notes are incomplete, outdated, or missing important
information needed to answer the user's request:
   - Set use_web=True.
   - Generate a list of concise and specific web search queries.No. of queries can depend on the amount of info required.

5. If the knowledge base of your own is insufficient to answer the question :
    - Set use_web = True
    - Generate a list of concise and specific web search queries.No. of queries can depend on the amount of info required.

Use web research only when it genuinely improves the quality or accuracy
of the final response.

Do NOT use web research simply because the topic exists online.

Do NOT answer the user's question.

Do NOT summarize the notes.

Do NOT teach the topic.

Your ONLY job is to fill the TeacherPlan schema.
"""


teacher_system = """
You are an expert AI Teacher.

Your goal is to teach the user in the clearest, most engaging, and most
accurate way possible.

You will be given:

1. The user's query.
2. Relevant excerpts retrieved from the user's uploaded notes (if any).
3. Relevant web research (if any).

Your job is to synthesize all available information into ONE coherent lesson.

Guidelines:

- Prioritize the user's uploaded notes whenever they contain the requested
  information.

- Use web research only to supplement missing, outdated, or incomplete
  information.

- Never contradict the uploaded notes unless the web research clearly shows
  that the notes are outdated or factually incorrect. If this happens,
  explicitly mention that newer information differs.

- Do not mention that you used RAG or retrieved chunks.

- Explain concepts step by step.

- Use simple language first, then gradually introduce technical depth.

- Define technical terms before using them.

- Whenever possible, include intuitive analogies or real-world examples.

- If the topic contains mathematics, derive equations step-by-step instead
  of only presenting the final formula.

- If code is involved:
    * explain each part
    * mention time complexity if relevant
    * explain why the solution works

- If the user requests a chapter or a broad topic, organize the response into
  sections with headings.

- If the user asks a short factual question, answer concisely.

- If the retrieved notes are insufficient and no web research is available,
  explicitly state which parts cannot be answered confidently.

- Never invent information that is not supported by either:
    * the uploaded notes,
    * the supplied web research,
    * or well-established general knowledge.

- Adapt your explanation to the apparent experience level of the user.
  If unsure, assume they are a university student.

Your objective is not merely to answer questions.

Your objective is to make the user genuinely understand the topic.
"""


TeachingPlanner = model.with_structured_output(TeacherPlan)


def teacher_node(state: AgentState):
    """Teaches people based on their uploaded notes if asked in the query or teaches based on the tool's own knowledge base. This tool handles all the questions asked and give answers"""
    

    teacher_planner_system = f"""
You are the resource planning component of an AI Teacher.

Your ONLY responsibility is to determine whether information from the user's
uploaded notes is required before teaching based on the given query of the user.

The user currently has access to the following uploaded files:

{state['uploaded_files']}

Follow these rules:

1. If the user explicitly asks:
   - "from my notes"
   - "from the uploaded pdf"
   - "from the slides"
   - "according to my notes"

   then use_rag MUST be True.

2. If the user's question can be answered accurately using general knowledge
and they do not request their uploaded notes, set use_rag=False.

3. If use_rag=True:
   - Choose the single most relevant filename from the uploaded files.
   - Generate a retrieval query that would retrieve the best chunks.
   - Decide how many chunks should be retrieved.

Chunk guide:
- Small definition/question → k = 5
- Explain a concept → k = 10
- Teach a topic → k = 20
- Large chapter/topic → k = 30
-Summarize -> Whole document's summary

Do NOT answer the user's question.

Do NOT explain the topic.

Your ONLY job is to fill the TeacherPlan schema.
"""
    rag_plan = TeachingPlanner.invoke([SystemMessage(content=teacher_planner_system),HumanMessage(content=state['query'])])
    docs =""
    if rag_plan.use_rag:
        docs = RAG_Tool(
            vectorestore=vectordb,
            query= rag_plan.rag_query,
            filename= rag_plan.filename,
            k= rag_plan.number_chunks
     
    )
    research_plan = TeachingPlanner.invoke([
    SystemMessage(content=web_research_system),
    HumanMessage(
        content=f"""
User Query:
{state['query']}

Retrieved Documents:
{docs}
"""
    )
])
    research_material = ""
    if research_plan.use_web:
        research_material = research_tool(queries=research_plan.web_queries)
    teacher = model.invoke([
    SystemMessage(content=teacher_system),
    HumanMessage(content=f"""
                 User Query:
                 {state['query']}
                 Retrieved Notes:
                 {docs}
                 Web Research:
                 {research_material}
""")
])
    return {"lesson":teacher.content,
            "current_task_index": state["current_task_index"] + 1}


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


def quiz_generator_node(
    state: AgentState
):
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
        HumanMessage(content=state["query"])
    ])

    docs = []
    docs_text = ""

    if quiz_plan.use_rag:

        docs = RAG_Tool(
            vectorestore=vectordb,
            query=quiz_plan.rag_query,
            filename=quiz_plan.filename,
            k=quiz_plan.number_chunks
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
""")
    ])
    return {
    "quiz": quiz,
    "current_task_index": state["current_task_index"] + 1
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
        )
    ])

    return {
        "quiz_evaluation": response,
        "current_task_index": state["current_task_index"] + 1
    }




def executor(state:AgentState):
    if state["current_task_index"] >= len(state["plan"].tasks):
        return END

    task = state["plan"].tasks[
        state["current_task_index"]
    ]

    return task.tool


graph = StateGraph(AgentState)

graph.add_node('planner',PlannerNode)
graph.add_node('teacher',teacher_node)
graph.add_node('quiz_generator',quiz_generator_node)
graph.add_node('quiz_eval',QuizEvaluationNode)



graph.add_edge(START,'planner')
graph.add_conditional_edges(
    "planner",
    executor,
    {
        "teacher": "teacher",
        "quiz_generator": "quiz_generator",
        "quiz_eval": "quiz_eval",
        END: END
    }
)

graph.add_edge(
    "teacher",
    "planner"
)

graph.add_edge(
    "quiz_generator",
    "planner"
)

graph.add_edge(
    "quiz_eval",
    "planner"
)

agent = graph.compile()
agent


