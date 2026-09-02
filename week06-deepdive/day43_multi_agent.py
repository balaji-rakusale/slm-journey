import json
import time
from datetime import datetime

print("=" * 50)
print("PART 1 - Multi Agent Architecture")
print("=" * 50)

print("""
Single Agent Limitation:
  One agent does everything
  Gets confused on complex tasks
  No specialization

Multi Agent Solution:
  Coordinator Agent → breaks task into subtasks
  Research Agent   → finds information
  Writer Agent     → writes content
  Reviewer Agent   → checks quality
  Each agent is specialized!

Real Example:
  Task: "Write a report on AI trends"

  Coordinator: splits into subtasks
  Research Agent: finds latest AI news
  Writer Agent: writes the report
  Reviewer Agent: checks accuracy
  Coordinator: combines and delivers
""")

print("=" * 50)
print("PART 2 - Build Specialized Agents")
print("=" * 50)


class BaseAgent:
    def __init__(self, name, role, tools=None):
        self.name = name
        self.role = role
        self.tools = tools or {}
        self.memory = []

    def think(self, task):
        return f"[{self.name}] Thinking about: {task}"

    def act(self, task):
        raise NotImplementedError

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{self.name}] {message}"
        self.memory.append(log_entry)
        print(log_entry)


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            role="Find and retrieve information"
        )
        self.knowledge_base = {
            "machine learning": "ML enables computers to learn from data automatically using statistical methods.",
            "deep learning": "Deep learning uses neural networks with multiple layers to learn complex patterns.",
            "transformer": "Transformers use attention mechanisms revolutionizing NLP since 2017.",
            "llm": "Large language models are trained on massive text data to generate human like text.",
            "rag": "RAG combines retrieval with generation for accurate grounded responses.",
            "lora": "LoRA reduces trainable parameters by 99% while maintaining model quality.",
            "agents": "AI agents combine reasoning and acting to complete complex multi step tasks.",
        }

    def act(self, query):
        self.log(f"Searching for: {query}")
        query_lower = query.lower()
        results = []

        for key, value in self.knowledge_base.items():
            if key in query_lower or any(
                    word in query_lower
                    for word in key.split()
            ):
                results.append(value)

        if results:
            result = " | ".join(results[:2])
            self.log(f"Found {len(results)} results")
            return result
        else:
            self.log("No results found")
            return "No information found"


class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="WriterAgent",
            role="Write clear structured content"
        )

    def act(self, task_with_context):
        self.log(f"Writing content...")
        parts = task_with_context.split("|||")
        topic = parts[0].strip()
        context = parts[1].strip() if len(parts) > 1 else ""

        output = f"""
# {topic.title()}

## Overview
{context}

## Key Points
- This is an important topic in modern AI
- Understanding this helps build better systems
- Practical applications are widespread

## Summary
{topic.title()} is fundamental to building 
modern AI systems effectively.
"""
        self.log("Content written successfully")
        return output.strip()


class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ReviewerAgent",
            role="Review and improve content quality"
        )

    def act(self, content):
        self.log("Reviewing content...")
        issues = []
        improvements = []

        if len(content) < 100:
            issues.append("Content too short")
            improvements.append("Add more detail")

        if "##" not in content:
            issues.append("Missing section headers")
            improvements.append("Add structured headers")

        if len(content.split()) < 30:
            issues.append("Insufficient word count")
            improvements.append("Expand explanations")

        score = 100 - (len(issues) * 20)
        score = max(score, 0)

        review = {
            "score": score,
            "issues": issues,
            "improvements": improvements,
            "approved": score >= 60
        }

        self.log(f"Review complete. Score: {score}/100")
        return review


class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CoordinatorAgent",
            role="Coordinate and delegate tasks"
        )
        self.research_agent = ResearchAgent()
        self.writer_agent = WriterAgent()
        self.reviewer_agent = ReviewerAgent()

    def act(self, task):
        self.log(f"Starting task: {task}")
        print()

        # Step 1 - Research
        self.log("Step 1: Delegating to ResearchAgent")
        research_result = self.research_agent.act(task)
        print()

        # Step 2 - Write
        self.log("Step 2: Delegating to WriterAgent")
        write_input = f"{task}|||{research_result}"
        written_content = self.writer_agent.act(write_input)
        print()

        # Step 3 - Review
        self.log("Step 3: Delegating to ReviewerAgent")
        review = self.reviewer_agent.act(written_content)
        print()

        # Step 4 - Final decision
        if review['approved']:
            self.log(f"✅ Content approved! Score: {review['score']}/100")
        else:
            self.log(f"⚠️ Content needs improvement. Score: {review['score']}/100")
            self.log(f"Issues: {review['issues']}")

        return {
            "task": task,
            "research": research_result,
            "content": written_content,
            "review": review,
            "status": "approved" if review['approved'] else "needs_revision"
        }


print("Agents initialized:")
coordinator = CoordinatorAgent()
print("  ✅ CoordinatorAgent")
print("  ✅ ResearchAgent")
print("  ✅ WriterAgent")
print("  ✅ ReviewerAgent")

print("\n" + "=" * 50)
print("PART 3 - Run Multi Agent Pipeline")
print("=" * 50)

tasks = [
    "machine learning fundamentals",
    "transformer architecture",
]

results = []
for task in tasks:
    print(f"\n{'=' * 45}")
    print(f"TASK: {task}")
    print('=' * 45)
    result = coordinator.act(task)
    results.append(result)
    print(f"\nFinal Status: {result['status'].upper()}")
    print(f"Content preview: {result['content'][:150]}...")

print("\n" + "=" * 50)
print("PART 4 - Agent Communication Patterns")
print("=" * 50)

print("""
Agent Communication Patterns:

1. Sequential (what we built):
   A → B → C → D
   Each agent passes to next
   Simple and predictable

2. Hierarchical:
   Coordinator at top
   Workers below
   Coordinator makes decisions

3. Parallel:
   Multiple agents run simultaneously
   Results combined at end
   Faster for independent tasks

4. Debate:
   Two agents argue different sides
   Third agent judges
   Good for decision making

5. Competitive:
   Multiple agents propose solutions
   Best solution selected
   Good for optimization tasks
""")

print("=" * 50)
print("PART 5 - Save Results")
print("=" * 50)

output = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "tasks_completed": len(results),
    "results": [
        {
            "task": r['task'],
            "status": r['status'],
            "score": r['review']['score']
        }
        for r in results
    ]
}

with open('multi_agent_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved ✅")
print(f"Tasks completed: {len(results)}")
for r in results:
    print(f"  → {r['task']}: {r['status']} (score: {r['review']['score']}/100)")

print("\nDay 43 Complete ✅")
print("Tomorrow: Fine tuning on custom domain data!")