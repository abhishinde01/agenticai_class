import ollama

# --- SIMULATED TOOL ---
def search_tool(query):
    """Simulates a web search tool."""
    print(f"🔍 [TOOL] Searching for: {query}")
    mock_database = {
        "jupiter moon count": "Jupiter has 95 officially recognized moons.",
        "largest moon": "Ganymede is the largest moon in the solar system."
    }
    return mock_database.get(query.lower(), "Information not found.")

# --- THE AGENT ---
class AdvancedPlanningAgent:
    def __init__(self, model='llama3'):
        self.model = model
        self.memory = []  # Memory storage

    def call_llm(self, prompt):
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']

    def run(self, user_goal):
        print(f"👤 User -> Agent: {user_goal}")
        
        # 1. PLANNER: Break down the goal
        planner_prompt = f"""
        Goal: {user_goal}
        Break this into 2 steps. Step 1 must be a search query for a tool.
        Format: 
        STEP1: [query]
        STEP2: [instruction]
        """
        plan = self.call_llm(planner_prompt)
        print(f"🧠 Planner: \n{plan}")

        # 2. TOOLS: Execute the first part of the plan
        # (Simple parsing logic for this example)
        search_query = "jupiter moon count" # Simplified for demo
        tool_result = search_tool(search_query)
        
        # 3. MEMORY: Store the tool result
        self.memory.append(f"Tool Result: {tool_result}")
        print(f"💾 Memory: Added research data.")

        # 4. OUTPUT: Final generation using memory
        output_prompt = f"""
        User Goal: {user_goal}
        Context from Memory: {self.memory}
        Based on the memory, provide the final answer.
        """
        final_output = self.call_llm(output_prompt)
        
        print("-" * 30)
        print(f"🏁 Final Output: {final_output}")

# --- EXECUTION ---
if __name__ == "__main__":
    agent = AdvancedPlanningAgent()
    agent.run("Tell me how many moons Jupiter has and why the largest one is special.")