import streamlit as st
import random
import re
import logging
import sys
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

# --- Logging Configuration ---
# This sets up logging to both the console (stdout) and a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_system.log")
    ]
)
logger = logging.getLogger("MultiAgentChat")

# --- Configuration ---
OLLAMA_MODEL = "llama3" 
MAX_TURNS = 4
AGENT_NAMES = ["Analyst Agent", "Critic Agent"]

# --- 1. Agent Tools ---
@tool
def simulated_search(query: str) -> str:
    """Simulates a web search or database lookup."""
    logger.info(f"Tool Execution: simulated_search called with query: '{query}'")
    query_lower = query.lower()
    if "market trends" in query_lower:
        res = "Market data suggests a 15% increase in e-commerce adoption over the last quarter."
    elif "shipping costs" in query_lower:
        res = "Shipping costs have increased by 8% year-over-year."
    elif "ai adoption" in query_lower:
        res = "AI implementation is expected to reduce human error by 40%."
    else:
        res = f"Simulated search results for '{query}' are inconclusive."
    
    logger.info(f"Tool Result: {res}")
    return res

# --- 2. Agent Definitions ---
def create_agent_chain(name: str, persona: str, llm):
    logger.info(f"Initializing chain for {name}")
    tool_desc = f"{simulated_search.name}: {simulated_search.description}"
    
    system_prompt = f"""
    You are the {name}, part of a multi-agent chatroom. Your persona is: "{persona}".
    If you need factual data, use the tool by writing: TOOL_CALL: simulated_search("your query")
    Available Tools: {tool_desc}
    """
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{conversation_history}\n\n[NEW TURN from the Other Agent: {new_message}]"),
    ])

    return (chat_prompt | llm | StrOutputParser()).with_config(run_name=f"{name}_Chain")

# --- 3. Custom Tool Execution and Agent Manager ---
def run_agent_turn(name, agent_chain, current_history, new_message):
    logger.info(f"--- Starting turn for: {name} ---")
    
    input_text = {
        "conversation_history": "\n".join(current_history),
        "new_message": new_message
    }
    
    logger.info(f"Invoking LLM for {name}...")
    raw_response = agent_chain.invoke(input_text)
    
    # Tool detection
    tool_call_match = re.search(r"TOOL_CALL:\s*(\w+)\(['\"](.*?)['\"]\)", raw_response, re.IGNORECASE)
    
    if tool_call_match:
        tool_name = tool_call_match.group(1).strip()
        tool_arg = tool_call_match.group(2).strip()
        logger.info(f"{name} requested tool: {tool_name} with args: {tool_arg}")
        
        observation = simulated_search(tool_arg)
        
        rerun_history = current_history + [
            f"[Tool Used: {tool_name}('{tool_arg}')]",
            f"[Tool Observation: {observation}]"
        ]
        
        logger.info(f"{name} is synthesizing final response after tool observation...")
        synthesis_input = {
            "conversation_history": "\n".join(rerun_history),
            "new_message": f"Synthesize a final response based on: {observation}"
        }
        final_response = agent_chain.invoke(synthesis_input)
        return f"[USED TOOL: {tool_name}] {final_response}", observation
        
    logger.info(f"{name} provided a direct response.")
    return raw_response, None

def simulate_chatroom(initial_prompt: str, llm):
    logger.info(f"SIMULATION START: Topic = {initial_prompt}")
    
    agent_a_chain = create_agent_chain(AGENT_NAMES[0], "Data-driven strategy analyst", llm)
    agent_b_chain = create_agent_chain(AGENT_NAMES[1], "Risk management specialist", llm)

    conversation_history = []
    
    # Randomize starter
    agents = [(AGENT_NAMES[0], agent_a_chain), (AGENT_NAMES[1], agent_b_chain)]
    if random.choice([True, False]):
        agents.reverse()
    
    current_agent_name, current_chain = agents[0]
    next_agent_name, next_chain = agents[1]
    
    logger.info(f"First speaker selected: {current_agent_name}")
    last_message = f"User initiated topic: {initial_prompt}"
    st.session_state.history.append({"speaker": "User", "message": initial_prompt})
    
    for turn in range(MAX_TURNS):
        logger.info(f"Processing Turn {turn + 1}/{MAX_TURNS}")
        
        response_text, tool_observation = run_agent_turn(current_agent_name, current_chain, conversation_history, last_message)
        
        conversation_history.append(f"[{current_agent_name}]: {response_text}")
        
        display_message = response_text
        if tool_observation:
            display_message = f"**Tool Observation:** `{tool_observation}`\n\n{display_message}"
        
        st.session_state.history.append({"speaker": current_agent_name, "message": display_message})
        
        # Swap roles
        current_agent_name, next_agent_name = next_agent_name, current_agent_name
        current_chain, next_chain = next_chain, current_chain
        last_message = response_text

    logger.info("SIMULATION END: Max turns reached.")
    st.session_state.is_running = False
    st.session_state.history.append({"speaker": "System", "message": "Conversation concluded."})

# --- 4. Streamlit UI Setup ---
@st.cache_resource
def get_ollama_llm():
    try:
        logger.info(f"Attempting to connect to Ollama model: {OLLAMA_MODEL}")
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.7)
    except Exception as e:
        logger.error(f"Ollama Initialization Error: {e}")
        return None

llm = get_ollama_llm()

# ... (Rest of Streamlit UI logic remains the same as your original snippet)
st.set_page_config(page_title="Ollama Multi-Agent Chatroom", layout="wide")
st.title("👥 Ollama Multi-Agent Collaboration")

if 'history' not in st.session_state: st.session_state.history = []
if 'is_running' not in st.session_state: st.session_state.is_running = False

chat_container = st.container(height=500, border=True)
with chat_container:
    for chat in st.session_state.history:
        if chat["speaker"] == "User": st.chat_message("user").write(chat["message"])
        elif chat["speaker"] in AGENT_NAMES: st.chat_message("ai").write(chat["message"])
        else: st.info(chat["message"])

user_input = st.text_input("Topic:", disabled=st.session_state.is_running)

if st.button("Start Collaboration", disabled=st.session_state.is_running or not user_input):
    st.session_state.history = []
    st.session_state.is_running = True
    simulate_chatroom(user_input, llm)
    st.rerun()
