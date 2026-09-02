import json
import re
from datetime import datetime

print("=" * 50)
print("PART 1 - What is Function Calling")
print("=" * 50)

print("""
Without Function Calling:
  User: "What is 15% tip on $85?"
  LLM:  "The tip would be around $12-13"
  Problem: Approximate, not exact!

With Function Calling:
  User: "What is 15% tip on $85?"
  LLM:  CALLS calculate(85 * 0.15)
  Tool: Returns 12.75
  LLM:  "The tip is exactly $12.75"

Function calling = LLM decides WHEN and HOW
to call external functions precisely.

Use cases:
  → Precise calculations
  → Real time data (weather, stocks)
  → Database queries
  → API calls
  → File operations
""")

print("=" * 50)
print("PART 2 - Define Functions")
print("=" * 50)

# Function definitions in JSON schema format
# Same format used by OpenAI, Anthropic, Ollama
function_definitions = [
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_database",
        "description": "Search company database for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "table": {
                    "type": "string",
                    "enum": ["products", "customers", "orders"],
                    "description": "Database table to search"
                }
            },
            "required": ["query", "table"]
        }
    },
]

print(f"Defined {len(function_definitions)} functions:")
for func in function_definitions:
    params = list(func['parameters']['properties'].keys())
    print(f"  → {func['name']}({', '.join(params)})")
    print(f"     {func['description']}")

print("\n" + "=" * 50)
print("PART 3 - Implement Functions")
print("=" * 50)


def calculate(expression: str) -> dict:
    try:
        allowed = set('0123456789+-*/()., ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return {"result": result, "expression": expression}
        return {"error": "Invalid expression"}
    except Exception as e:
        return {"error": str(e)}


def get_weather(city: str, unit: str = "celsius") -> dict:
    # Simulated weather data
    weather_data = {
        "pune": {"temp_c": 28, "condition": "Partly cloudy", "humidity": 65},
        "mumbai": {"temp_c": 31, "condition": "Humid", "humidity": 80},
        "delhi": {"temp_c": 35, "condition": "Sunny", "humidity": 40},
        "bangalore": {"temp_c": 24, "condition": "Pleasant", "humidity": 70},
    }

    city_lower = city.lower()
    data = weather_data.get(city_lower, {
        "temp_c": 25,
        "condition": "Clear",
        "humidity": 60
    })

    temp = data['temp_c']
    if unit == "fahrenheit":
        temp = (temp * 9 / 5) + 32

    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": data['condition'],
        "humidity": data['humidity']
    }


def search_database(query: str, table: str) -> dict:
    # Simulated database
    database = {
        "products": [
            {"id": 1, "name": "AI Assistant Pro", "price": 99},
            {"id": 2, "name": "ML Pipeline Tool", "price": 199},
            {"id": 3, "name": "RAG Builder", "price": 149},
        ],
        "customers": [
            {"id": 1, "name": "Tech Corp", "plan": "Enterprise"},
            {"id": 2, "name": "StartupXYZ", "plan": "Starter"},
        ],
        "orders": [
            {"id": 1, "customer": "Tech Corp", "product": "AI Assistant Pro", "status": "active"},
            {"id": 2, "customer": "StartupXYZ", "product": "RAG Builder", "status": "trial"},
        ]
    }

    results = []
    query_lower = query.lower()
    for item in database.get(table, []):
        item_str = json.dumps(item).lower()
        if any(word in item_str for word in query_lower.split()):
            results.append(item)

    return {
        "query": query,
        "table": table,
        "results": results,
        "count": len(results)
    }


# Function registry
FUNCTIONS = {
    "calculate": calculate,
    "get_weather": get_weather,
    "search_database": search_database,
}

print("Testing functions:")
print(f"calculate('85 * 0.15'):     {calculate('85 * 0.15')}")
print(f"get_weather('Pune'):         {get_weather('Pune')}")
print(f"search_database('AI', 'products'): {search_database('AI', 'products')}")

print("\n" + "=" * 50)
print("PART 4 - Function Call Parser")
print("=" * 50)


def parse_function_call(llm_output: str) -> dict:
    pattern = r'FUNCTION_CALL:\s*(\w+)\((.*?)\)'
    match = re.search(pattern, llm_output, re.DOTALL)

    if not match:
        return None

    func_name = match.group(1)
    args_str = match.group(2)

    try:
        args = json.loads(f"{{{args_str}}}")
    except:
        args = {"input": args_str.strip("'\"")}

    return {
        "function": func_name,
        "arguments": args
    }


def execute_function_call(func_call: dict) -> str:
    if not func_call:
        return "No function call found"

    func_name = func_call['function']
    args = func_call['arguments']

    if func_name not in FUNCTIONS:
        return f"Function {func_name} not found"

    result = FUNCTIONS[func_name](**args)
    return json.dumps(result)


# Simulate LLM outputs with function calls
simulated_llm_outputs = [
    'I need to calculate this. FUNCTION_CALL: calculate("expression": "85 * 0.15")',
    'Let me check the weather. FUNCTION_CALL: get_weather("city": "Pune", "unit": "celsius")',
    'Searching the database. FUNCTION_CALL: search_database("query": "AI", "table": "products")',
]

print("Parsing function calls from LLM output:")
for output in simulated_llm_outputs:
    func_call = parse_function_call(output)
    if func_call:
        result = execute_function_call(func_call)
        print(f"\nLLM Output: {output[:60]}...")
        print(f"Parsed:     {func_call}")
        print(f"Result:     {result[:100]}")

print("\n" + "=" * 50)
print("PART 5 - Complete Function Calling Pipeline")
print("=" * 50)


class FunctionCallingAssistant:
    def __init__(self, functions, function_defs):
        self.functions = functions
        self.function_defs = function_defs
        self.conversation = []

    def process_query(self, user_query):
        print(f"\nUser: {user_query}")

        # Simple rule based function selection
        # In production LLM decides this
        selected_func = None
        args = {}

        if any(op in user_query for op in ['+', '-', '*', '/', '%', 'calculate', 'tip', 'percent']):
            numbers = re.findall(r'\d+\.?\d*', user_query)
            if len(numbers) >= 2:
                selected_func = "calculate"
                args = {"expression": f"{numbers[0]} * {numbers[1]} / 100"
                if 'percent' in user_query or '%' in user_query or 'tip' in user_query
                else f"{numbers[0]} + {numbers[1]}"}

        elif 'weather' in user_query.lower():
            cities = ['pune', 'mumbai', 'delhi', 'bangalore']
            for city in cities:
                if city in user_query.lower():
                    selected_func = "get_weather"
                    args = {"city": city}
                    break
            if not selected_func:
                selected_func = "get_weather"
                args = {"city": "pune"}

        elif any(word in user_query.lower() for word in ['find', 'search', 'show', 'list']):
            selected_func = "search_database"
            table = "products"
            for t in ["products", "customers", "orders"]:
                if t in user_query.lower():
                    table = t
            args = {"query": user_query, "table": table}

        if selected_func:
            print(f"Calling: {selected_func}({args})")
            result = self.functions[selected_func](**args)
            print(f"Result:  {json.dumps(result)[:100]}")
            return result
        else:
            print("No function needed — answering directly")
            return {"answer": "I can help with calculations, weather, and database searches!"}


# Test the assistant
assistant = FunctionCallingAssistant(FUNCTIONS, function_definitions)

queries = [
    "What is 15 percent of 85?",
    "What is the weather in Pune?",
    "Find all products in database",
    "Hello how are you?",
]

print("Testing Function Calling Assistant:")
for query in queries:
    assistant.process_query(query)

print("\n" + "=" * 50)
print("PART 6 - Structured Output")
print("=" * 50)

print("""
Structured Output = forcing LLM to output
valid JSON instead of free text

Why important:
  Free text output → hard to parse
  Structured JSON  → easy to use in code

Example:

Without structured output:
  "The sentiment is positive and the
   confidence is around 95 percent"
  Hard to parse!

With structured output:
  {
    "sentiment": "positive",
    "confidence": 0.95,
    "keywords": ["great", "love", "excellent"]
  }
  Easy to use in code!

How to implement:
  1. Tell LLM to output JSON only
  2. Provide JSON schema in prompt
  3. Validate output against schema
  4. Retry if invalid JSON
""")

# Example structured output schema
sentiment_schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["sentiment", "confidence", "keywords"]
}

# Simulate structured outputs
test_texts = [
    "I absolutely love this product! Amazing quality!",
    "This is terrible, worst purchase ever.",
    "It works okay, nothing special.",
]


def simple_sentiment(text):
    positive_words = {'love', 'amazing', 'great', 'excellent', 'good', 'best', 'wonderful'}
    negative_words = {'terrible', 'worst', 'bad', 'awful', 'horrible', 'poor'}

    words = set(text.lower().split())
    pos_count = len(words.intersection(positive_words))
    neg_count = len(words.intersection(negative_words))

    if pos_count > neg_count:
        sentiment = "positive"
        confidence = min(0.95, 0.6 + pos_count * 0.1)
        keywords = list(words.intersection(positive_words))
    elif neg_count > pos_count:
        sentiment = "negative"
        confidence = min(0.95, 0.6 + neg_count * 0.1)
        keywords = list(words.intersection(negative_words))
    else:
        sentiment = "neutral"
        confidence = 0.6
        keywords = []

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 2),
        "keywords": keywords
    }


print("Structured output examples:")
for text in test_texts:
    result = simple_sentiment(text)
    print(f"\nText:   {text}")
    print(f"Output: {json.dumps(result)}")

print("\nDay 42 Complete ✅")
print("Now continue to Day 43!")