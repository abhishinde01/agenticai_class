import ollama

def call_llm(prompt):
    response = ollama.generate(model='llama3', prompt=prompt)
    return response['response']

def planning_agent(user_goal):
    print(f"🚀 Goal: {user_goal}\n")

    # --- STAGE 1: PLANNING ---
    print("🧠 Thinking... Creating a plan.")
    planner_prompt = f"""
    You are a planning assistant. Break down the following goal into a numbered list of 3 logical steps.
    Goal: {user_goal}
    Response format:
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]
    """
    plan = call_llm(planner_prompt)
    print(f"📋 THE PLAN:\n{plan}\n")

    # --- STAGE 2: EXECUTION ---
    print("🛠 Executing the plan...")
    execution_prompt = f"""
    You are an expert executor. I will give you a goal and a plan. 
    Complete the goal by following the plan steps exactly.
    
    GOAL: {user_goal}
    PLAN:
    {plan}
    
    FINAL OUTPUT:
    """
    final_result = call_llm(execution_prompt)
    
    print("✅ EXECUTION COMPLETE:\n")
    print(final_result)

# Run the agent
if __name__ == "__main__":
    topic = "Write a short educational paragraph about how black holes evaporate."
    planning_agent(topic)