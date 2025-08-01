########################################
# 🚀 1. Import Required Libraries
########################################

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import io

from typing import Annotated, TypedDict
from PIL import Image

# LangChain & LangGraph Imports
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


########################################
# 🔐 2. Load Environment Variables
########################################

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

########################################
# 🌐 3. Initialize FastAPI App & Enable CORS
########################################

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
# 📦 4. Define Input Schema for Requests
########################################

class ChatRequest(BaseModel):
    message: str
    session_id: str

########################################
# 🧰 5. Define External Tools
########################################

# Wikipedia Tool
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
)

# Python REPL Tool
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Execute Python code using this shell. Use print(...) to get results.",
    func=python_repl.run,
)

########################################
# 🤖 6. Initialize LLM and Bind Tools
########################################

llm = ChatGroq(api_key=groq_api_key, model="llama3-70b-8192", temperature=0)
llm_with_tools = llm.bind_tools(tools=[wiki_tool, repl_tool])

########################################
# 📡 7. Define Graph State Format
########################################

class State(TypedDict):
    messages: Annotated[list, add_messages]

########################################
# 🧠 8. Build LangGraph Workflow
########################################

graph_builder = StateGraph(State)

# Step 1: Call LLM with possible tool-use
graph_builder.add_node("chatbot", lambda state: {
    "messages": [llm_with_tools.invoke(state["messages"])]
})

# Step 2: Tool handler node
graph_builder.add_node("tools", ToolNode(tools=[wiki_tool, repl_tool]))

# Step 3: Define flow edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

########################################
# 🖼️ 9. Save Graph Visualization as PNG
########################################

image_data = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(image_data))
image.save("langgraph_visualization.png")

########################################
# 💬 10. Chat Endpoint to Process User Message
########################################

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    question = request.message

    # Add initial system instruction
    system_prompt = """
    You are an AI assistant powered by LangGraph and Groq's LLaMA 3.
    Capabilities:
    - 🔍 Use Wikipedia to answer factual questions.
    - 🧮 Use Python REPL for computations and code execution.

    - Answer questions using Wikipedia or Python when needed.
    - Greet and explain capabilities only once on first interaction.
    - For all other responses:
      - No greetings, no fluff.
      - Respond concisely, factually, and only to the question.
      - Prefer short sentences and bullet points if applicable.
    """

    # Stream LangGraph events
    events = graph.stream(
        {"messages": [("system", system_prompt), ("user", question)]},
        stream_mode="values"
    )

    # Extract response
    response = ""
    for event in events:
        response = event["messages"][-1].content
        for m in event["messages"]:
            m.pretty_print()
    return {"response": response}

########################################
# 🏠 11. Serve Static Frontend HTML (UI)
########################################

@app.get("/")
async def serve_html():
    return FileResponse("chatbot_ui.html")