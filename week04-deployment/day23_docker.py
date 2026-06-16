import subprocess
import json

print("=" * 50)
print("PART 1 - What is Docker")
print("=" * 50)

docker_concepts = {
    "Image": "Blueprint of your app — like a recipe",
    "Container": "Running instance of image — like a cooked meal",
    "Dockerfile": "Instructions to build image — like a recipe card",
    "Registry": "Store for images — like a recipe book",
    "Port": "Door number your app listens on",
}

print("Docker Concepts:")
for concept, explanation in docker_concepts.items():
    print(f"  {concept:12} → {explanation}")

print("\n" + "=" * 50)
print("PART 2 - Why Docker For ML Deployment")
print("=" * 50)

problems_without_docker = [
    "Works on my machine but not server",
    "Different Python versions cause errors",
    "Missing dependencies crash the app",
    "Hard to scale to multiple servers",
    "No isolation between services",
]

solutions_with_docker = [
    "Same container runs everywhere",
    "Python version locked in Dockerfile",
    "All dependencies packaged together",
    "Scale by running more containers",
    "Each service in its own container",
]

print(f"{'Without Docker':<40} {'With Docker'}")
print("-" * 75)
for prob, sol in zip(problems_without_docker, solutions_with_docker):
    print(f"❌ {prob:<38} ✅ {sol}")

print("\n" + "=" * 50)
print("PART 3 - Docker Commands Cheat Sheet")
print("=" * 50)

commands = [
    ("docker build -t slm-api .", "Build image from Dockerfile"),
    ("docker run -p 8000:8000 slm-api", "Run container"),
    ("docker ps", "List running containers"),
    ("docker stop <id>", "Stop container"),
    ("docker images", "List all images"),
    ("docker logs <id>", "View container logs"),
    ("docker push slm-api", "Push to registry"),
]

print("Essential Docker commands:")
for cmd, desc in commands:
    print(f"  {cmd:<40} → {desc}")

print("\n" + "=" * 50)
print("PART 4 - Production Architecture")
print("=" * 50)

print("""
Enterprise Deployment Architecture:

Client Request
      ↓
Load Balancer (nginx)
      ↓
┌─────────────────────────┐
│  Docker Container 1     │
│  SLM API (port 8000)    │
├─────────────────────────┤
│  Docker Container 2     │
│  SLM API (port 8001)    │
├─────────────────────────┤
│  Docker Container 3     │
│  SLM API (port 8002)    │
└─────────────────────────┘
      ↓
GPU Server / Cloud

This handles thousands of requests per second.
Each container is isolated and scalable.
One container crashes — others keep running.
""")

print("\n" + "=" * 50)
print("PART 5 - Build and Run Docker")
print("=" * 50)

print("""
Run these commands in your terminal:

# Step 1 - Build image
cd week04-deployment
docker build -t slm-api .

# Step 2 - Run container
docker run -p 8000:8000 slm-api

# Step 3 - Test
curl http://localhost:8000/health

# Step 4 - Stop
docker stop $(docker ps -q)
""")

print("\n" + "=" * 50)
print("PART 6 - Cloud Deployment Options")
print("=" * 50)

cloud_options = [
    {
        "name": "Hugging Face Spaces",
        "cost": "Free",
        "difficulty": "Easy",
        "best_for": "Demo and portfolio"
    },
    {
        "name": "Railway.app",
        "cost": "$5/month",
        "difficulty": "Easy",
        "best_for": "Small production apps"
    },
    {
        "name": "AWS EC2 + GPU",
        "cost": "$0.5-3/hr",
        "difficulty": "Medium",
        "best_for": "Enterprise production"
    },
    {
        "name": "Google Cloud Run",
        "cost": "Pay per request",
        "difficulty": "Medium",
        "best_for": "Scalable APIs"
    },
    {
        "name": "Azure ML",
        "cost": "Variable",
        "difficulty": "Hard",
        "best_for": "Enterprise with Azure"
        },
]

print(f"{'Platform':<25} {'Cost':<15} {'Difficulty':<12} {'Best For'}")
print("-" * 70)
for opt in cloud_options:
    print(f"{opt['name']:<25} {opt['cost']:<15} {opt['difficulty']:<12} {opt['best_for']}")