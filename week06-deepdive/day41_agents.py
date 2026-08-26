import json
import requests
import time
import math
from datetime import datetime

print("=" * 50)
print("PART 1 - What Are AI Agents")
print("=" * 50)

print("""
Standard LLM (What we built):
  User: "What is the weather in Pune?"
  LLM:  "I don't know current weather."
  Done. Dead end.

AI Agent:
  User:  "What is the weather in Pune?"
  Agent: THINK → I need to search for weather
  Agent: ACT   → call weather API
  Agent: OBS   → temperature is 28C, sunny
  Agent: THINK → I have the answer now
  Agent: RESP  → "It is 28C and sunny in Pune!"

Agent Loop:
  Think → Act → Observe → Think → Act → Observe...
  Until task is complete

This is called ReAct:
  RE = Reasoning
  Act = Acting
""")

print("=" * 50)
print("PART 2 - Build Tool Functions")
print("=" * 50)

# Tools the agent can use
def calculator(expression: str) -> str:
    try:
        # Safe evaluation of math expressions
        allowed = set('0123456789+-*/()., ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        return "Error: Invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_time() -> str:
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def search_knowledge_base(query: str) -> str:
    knowledge = {
        "python": "Python is a high level programming language known for simplicity.",
        "machine learning": "Machine learning enables computers to learn from data automatically.",
        "deep learning": "Deep learning uses neural networks with many layers.",
        "transformer": "Transformers use attention mechanisms to process sequences.",
        "lora": "LoRA trains small adapter matrices instead of full model weights.",
        "rag": "RAG combines retrieval with generation for accurate answers.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f"Found: {value}"
    return "Not found in knowledge base"

def word_counter(text: str) -> str:
    words = len(text.split())
    chars = len(text)
    return f"Words: {words}, Characters: {chars}"

# Tool registry
TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "Calculate mathematical expressions. Input: math expression as string.",
        "example": "calculator('2 + 2 * 10')"
    },
    "get_time": {
        "function": get_current_time,
        "description": "Get current date and time. No input needed.",
        "example": "get_time()"
    },
    "search_kb": {
        "function": search_knowledge_base,
        "description": "Search knowledge base for AI/ML topics. Input: search query.",
        "example": "search_kb('what is machine learning')"
    },
    "word_counter": {
        "function": word_counter,
        "description": "Count words and characters in text. Input: text string.",
        "example": "word_counter('Hello world this is a test')"
    },
}

print("Available tools:")
for name, tool in TOOLS.items():
    print(f"  → {name}: {tool['description'][:60]}")

print("\n" + "=" * 50)
print("PART 3 - ReAct Agent Implementation")
print("=" * 50)

class ReActAgent:
    def __init__(self, tools, max_steps=5):
        self.tools = tools
        self.max_steps = max_steps

    def build_system_prompt(self):
        tool_descriptions = "\n".join([
            f"- {name}: {tool['description']}"
            for name, tool in self.tools.items()
        ])
        return f"""You are an AI agent that solves tasks step by step.

Available tools:
{tool_descriptions}

Format your response as:
THOUGHT: your reasoning about what to do next
ACTION: tool_name('input')
OBSERVATION: (this will be filled by the system)
... repeat as needed ...
FINAL ANSWER: your final response to the user"""

    def parse_action(self, text):
        lines = text.strip().split('\n')
        for line in lines:
            if line.startswith('ACTION:'):
                action = line.replace('ACTION:', '').strip()
                for tool_name in self.tools:
                    if action.startswith(tool_name):
                        try:
                            input_start = action.index('(') + 1
                            input_end = action.rindex(')')
                            tool_input = action[input_start:input_end].strip("'\"")
                            return tool_name, tool_input
                        except:
                            return tool_name, ""
        return None, None

    def execute_tool(self, tool_name, tool_input):
        if tool_name not in self.tools:
            return f"Error: Tool {tool_name} not found"

        tool_func = self.tools[tool_name]['function']
        try:
            if tool_input:
                return tool_func(tool_input)
            else:
                return tool_func()
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def run(self, task):
        print(f"\nTask: {task}")
        print("-" * 40)

        # Simulate agent reasoning
        # In production this would call your LLM
        steps = []
        step_count = 0

        # Simple rule based agent for demonstration
        if any(op in task for op in ['+', '-', '*', '/', 'calculate', 'math']):
            thought = "I need to calculate a mathematical expression"
            # Extract numbers from task
            import re
            expression = re.findall(r'[\d\+\-\*\/\(\)\. ]+', task)
            expression = expression[0].strip() if expression else "1+1"
            action = f"calculator('{expression}')"
            tool_name, tool_input = "calculator", expression
            observation = self.execute_tool(tool_name, tool_input)
            steps.append({"thought": thought, "action": action, "observation": observation})

        elif "time" in task.lower() or "date" in task.lower():
            thought = "I need to get the current time"
            action = "get_time()"
            observation = self.execute_tool("get_time", "")
            steps.append({"thought": thought, "action": action, "observation": observation})

        elif any(word in task.lower() for word in ["what is", "explain", "define"]):
            thought = "I should search the knowledge base"
            action = f"search_kb('{task}')"
            observation = self.execute_tool("search_kb", task)
            steps.append({"thought": thought, "action": action, "observation": observation})

        elif "count" in task.lower() or "words" in task.lower():
            thought = "I need to count words in the text"
            text = task.replace("count words in", "").replace("how many words in", "").strip()
            action = f"word_counter('{text}')"
            observation = self.execute_tool("word_counter", text)
            steps.append({"thought": thought, "action": action, "observation": observation})

        # Print agent trace
        for i, step in enumerate(steps):
            print(f"THOUGHT {i+1}: {step['thought']}")
            print(f"ACTION {i+1}:  {step['action']}")
            print(f"OBS {i+1}:     {step['observation']}")
            print()

        final_answer = steps[-1]['observation'] if steps else "Could not complete task"
        print(f"FINAL ANSWER: {final_answer}")
        return final_answer

# Test the agent
agent = ReActAgent(TOOLS)

test_tasks = [
    "Calculate 15 * 24 + 100",
    "What is the current time and date?",
    "What is machine learning?",
    "Count words in: The quick brown fox jumps over the lazy dog",
]

print("Running ReAct Agent:")
for task in test_tasks:
    agent.run(task)
    print("=" * 40)

print("\n" + "=" * 50)
print("PART 4 - Agent Patterns")
print("=" * 50)

print("""
Common Agent Patterns:

1. ReAct (Reasoning + Acting)
   Most common pattern
   Think before each action
   Good for complex tasks

2. Plan and Execute
   Make full plan first
   Then execute each step
   Good for multi step tasks

3. Reflexion
   Agent reflects on mistakes
   Tries again with corrections
   Good for accuracy critical tasks

4. Multi Agent
   Multiple specialized agents
   Coordinator delegates tasks
   Good for complex workflows

Real World Agent Examples:
  → Code agent: writes and runs code
  → Research agent: searches and summarizes
  → Data agent: queries databases
  → Email agent: reads and drafts emails
""")

print("=" * 50)
print("PART 5 - Ollama Agent Integration")
print("=" * 50)

def ollama_agent_step(messages, model="phi3:mini"):
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200}
        }
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['message']['content']
        return None
    except:
        return None

# Check if Ollama is running
ollama_response = ollama_agent_step([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say 'Ollama connected' if you can read this."}
])

if ollama_response:
    print(f"Ollama connected ✅")
    print(f"Response: {ollama_response[:100]}")

    # Real LLM agent task
    agent_prompt = """You are an agent with these tools:
- calculator: for math
- get_time: for current time
- search_kb: for AI knowledge

Task: What is 42 * 8? Also what is machine learning?

Think step by step and use the appropriate tools.
Format: THOUGHT: ... ACTION: tool_name('input')"""

    response = ollama_agent_step([
        {"role": "user", "content": agent_prompt}
    ])

    if response:
        print(f"\nLLM Agent Response:")
        print(response[:300])
else:
    print("Ollama not running — start with: ollama serve")
    print("Rule-based agent demonstrated above instead")

print("\n" + "=" * 50)
print("PART 6 - Agent Memory")
print("=" * 50)

print("""
Agent Memory Types:

1. In-Context Memory (Short term)
   → Conversation history in prompt
   → Limited by context window
   → Lost when conversation ends

2. External Memory (Long term)
   → Store in vector database
   → Retrieve relevant memories
   → Persists across conversations
   → Use ChromaDB from Day 26!

3. Episodic Memory
   → Remember specific past events
   → "Last time user asked X we did Y"
   → Improves over time

4. Semantic Memory
   → Learned knowledge about user
   → Preferences, patterns, facts
   → Built from interaction history

For your RAG system:
  User asks question → search memory
  Find relevant past interactions
  Add to context → better answers
  This makes AI feel personalized
""")

print("Day 41 Complete ✅")
print("Tomorrow: Multi agent systems!")