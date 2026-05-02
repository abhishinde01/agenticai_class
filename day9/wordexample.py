import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup local knowledge from folder
DOCS_PATH = "C://ml//code//day9//my_word_files"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load and process files
loader = DirectoryLoader(DOCS_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(docs)

# Create Vector Store
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 2. Create the Tool
tool = create_retriever_tool(
    vectorstore.as_retriever(),
    "word_doc_search",
    "Searches the local folder for Word documents. Use this to answer specific questions about doc content."
)
tools = [tool]

# 3. THE "NEW" MANUAL PROMPT (Replaces hub.pull)
# ReAct agents require these exact 3 variables: {tools}, {tool_names}, and {agent_scratchpad}
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# 4. Initialize Ollama and Agent
llm = ChatOllama(model="llama3", temperature=0,streaming=True ) # <--- This makes the agent's "Thinking" appear live)

# Build the agent with the manual prompt
agent = create_react_agent(llm, tools, prompt)

# 5. Create Executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)

# Example Run


for chunk in agent_executor.stream({"input": "What is the deadline for Project Alpha?"}):
    # Each 'chunk' is a dictionary containing either actions or the final output
    if "actions" in chunk:
        for action in chunk["actions"]:
            print(f"DEBUG: Agent is calling tool: {action.tool}")
    elif "steps" in chunk:
        for step in chunk["steps"]:
            print(f"DEBUG: Tool returned: {step.observation[:50]}...")
    elif "output" in chunk:
        print(f"\n--- FINAL ANSWER ---\n{chunk['output']}")