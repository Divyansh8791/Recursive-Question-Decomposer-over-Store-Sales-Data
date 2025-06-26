###### Importing necessary libraries

from langchain_community.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
import pandas as pd
import io
import contextlib
from dotenv import load_dotenv
import pandas as pd


# importing cleaned data
df = pd.read_csv("cleaned_demo_sales_data.csv")

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


##### Langchain tool for generating answer for atomic query .

code_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a data assistant working with a pandas DataFrame called `df` that has the following columns:

['Store Name', 'Description', 'Department', 'Qty Sold', 'Cost', 'Retail', 'Total Retail', 'Margin', 'Profit']

Generate a Python print() statement using pandas that answers the following question:

Question: {question}

Return only valid Python code. Do not explain anything. Only return the code.

    """
)

code_chain = LLMChain(llm=llm, prompt=code_prompt)

# function to safely run code
def safe_run_pandas_query(code: str) -> str:
    local_vars = {"df": df.copy()}
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {}, local_vars)
        return stdout_capture.getvalue() or "Code executed successfully."
    except Exception as e:
        return f"Error executing code: {e}"

# LangChain Tool
def pandas_query_tool_fn(question: str) -> str:
    try:
        # Get code from LLM
        code = code_chain.run(question)
        # print("Generated code:\n", code) 

        # Clean code if wrapped in markdown ``` blocks
        if "```" in code:
            code = code.split("```")[1]  
        code = code.replace("python", "").strip() 

        # Execute the cleaned code
        result = safe_run_pandas_query(code)
        return result.strip()
    except Exception as e:
        return f"Error: {e}"

# Create a LangChain Tool object
pandas_query_tool = Tool(
    name="Pandas Query Tool",
    func=pandas_query_tool_fn,
    description=(
        "Use this tool to run pandas-based queries on store sales data. "
        "You must provide a valid Python code string that uses the dataframe `df`. "
        "Only use this for atomic data-resolvable questions."
    ),
    return_direct=True,
)

answer = pandas_query_tool.run("What is the total profit from HOT FOOD?")
# print(answer)



################ Langraph structure and state structure 

from typing import TypedDict, List, Optional, Union

class QuestionState(TypedDict):
    original_question: str
    current_question: str
    is_complex: Optional[bool]
    sub_questions: List[str]
    answers: List[dict]
    reasoning_trace: List[str]
    final_answer: Optional[str]

def initialize_state(user_question: str) -> QuestionState:
    return {
        "original_question": user_question,
        "current_question": user_question,
        "is_complex": None,
        "sub_questions": [],
        "answers": [],
        "reasoning_trace": [],
        "final_answer": None,
    }


########## Defining Complexity node which will tell if the query is complex or not .

from langgraph.graph import StateGraph, END

complexity_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert at analyzing natural language questions.

Determine if the following question is complex, meaning it has multiple sub-parts or requires multiple steps to answer:

Question: {question}

Answer only "yes" or "no".
"""
)

# Chain
complexity_chain = LLMChain(llm=llm, prompt=complexity_prompt)

# Node Function
def complexity_node(state: QuestionState) -> QuestionState:
    question = state["current_question"]
    decision = complexity_chain.run(question).strip().lower()

    is_complex = decision.startswith("y")  # yes -> True, no -> False

    # Update state
    state["is_complex"] = is_complex
    state["reasoning_trace"].append(f"Classified question as {'complex' if is_complex else 'atomic'}.")

    return state


######## Creating Decomposing node which will decompose the complex query into sub-queries : 


# Prompt for Decomposition
decompose_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an AI assistant that breaks down complex analytical questions into smaller, simple, atomic questions.

Decompose the following question into the smallest number of clear, data-based sub-questions needed to fully answer it.

Question: {question}

Return the sub-questions as a numbered list. Do not explain anything else.
"""
)

# Chain
decompose_chain = LLMChain(llm=llm, prompt=decompose_prompt)

# Node Function
def decomposition_node(state: QuestionState) -> QuestionState:
    question = state["current_question"]

    # Run LLM to get sub-questions
    raw_output = decompose_chain.run(question)

    # Extract sub-questions from numbered list
    sub_qs = [line.split('. ', 1)[-1].strip() for line in raw_output.strip().splitlines() if line.strip()]

    # Update state
    state["sub_questions"].extend(sub_qs)
    state["reasoning_trace"].append(f"Decomposed into sub-questions: {sub_qs}")

    return state


####### Creating data-resolver node , which will answer the multiple queries and updte the state : 

def data_resolver_node(state: QuestionState) -> QuestionState:
    # the next sub-question
    if not state["sub_questions"]:
        state["reasoning_trace"].append("No sub-questions left to resolve.")
        return state

    current_q = state["sub_questions"].pop(0)

    # Run query using your tool
    answer = pandas_query_tool_fn(current_q)

    # Store answer
    state["answers"].append({"question": current_q, "answer": answer})

    # Update reasoning trace
    state["reasoning_trace"].append(f"Answered: '{current_q}' → {answer}")

    # 5. Set current_question 
    state["current_question"] = current_q

    return state


#### Creating Aggregate node which will aggregate the final answer into json : 

import json

def aggregator_node(state: QuestionState) -> QuestionState:
    answers = state["answers"]

    # Build a structured JSON-style summary
    summary = {
        "original_question": state["original_question"],
        "sub_question_answers": answers,
        "reasoning_trace": state["reasoning_trace"]
    }

    # Save in final_answer as JSON string
    state["final_answer"] = json.dumps(summary, indent=2)

    # Add trace log
    state["reasoning_trace"].append("Aggregated all answers into final JSON output.")

    return state



######################## : Defining the graph :

from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessageGraph

graph = StateGraph(QuestionState)
# 3. Add nodes
graph.add_node("complexity_node", complexity_node)
graph.add_node("decomposition_node", decomposition_node)
graph.add_node("data_resolver_node", data_resolver_node)
graph.add_node("aggregator_node", aggregator_node)

# 4. Add edges and control flow
graph.set_entry_point("complexity_node")

graph.add_conditional_edges(
    "complexity_node",
    lambda state: "decomposition_node" if state["is_complex"] else "data_resolver_node"
)

graph.add_edge("decomposition_node", "data_resolver_node")

# Loop data resolver until all sub-questions are resolved
def should_continue(state: QuestionState):
    return "data_resolver_node" if state["sub_questions"] else "aggregator_node"

graph.add_conditional_edges("data_resolver_node", should_continue)

graph.add_edge("aggregator_node", END)

# Compile the LangGraph
app = graph.compile()

# Input main user question
user_q = "What are the top-selling products by revenue in each store?"
state = initialize_state(user_q)

# Run the graph
final_state = app.invoke(state)

# Show final output
# import json
# print(json.dumps(json.loads(final_state["final_answer"]), indent=2))

# print(final_state)