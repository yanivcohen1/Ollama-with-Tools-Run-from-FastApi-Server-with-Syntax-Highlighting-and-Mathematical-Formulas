import ollama
import json

# 1. Define the actual Python functions (Tools)
def get_weather(city: str) -> str:
    """Mock function to get the weather for a city."""
    print(f"    [Executing get_weather for {city}]")
    # In a real app, you would call a weather API here
    weather_data = {
        "haifa": "sunny and 25°C",
        "tel aviv": "cloudy and 20°C",
        "jerusalem": "windy and 18°C"
    }
    return weather_data.get(city.lower(), "unknown weather")

def recommend_activity(weather: str) -> str:
    """Mock function to recommend an activity based on weather."""
    print(f"    [Executing recommend_activity for weather: {weather}]")
    weather_lower = weather.lower()
    if "sunny" in weather_lower:
        return "Go to the beach or have a picnic!"
    elif "rainy" in weather_lower or "cloudy" in weather_lower:
        return "Visit a museum or go to a cafe."
    else:
        return "Read a book indoors or go for a walk."

# Map function names to the actual functions
available_tools = {
    "get_weather": get_weather,
    "recommend_activity": recommend_activity
}

# 2. Define the tool schemas for Ollama
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for, e.g. Haifa"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_activity",
            "description": "Recommend an activity based on the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "weather": {
                        "type": "string",
                        "description": "The weather condition, e.g. sunny and 25°C"
                    }
                },
                "required": ["weather"]
            }
        }
    }
]

def run_agent(prompt: str, model_name: str = "qwen3:1.7b"):
    print(f"\n--- Starting Agent with prompt: '{prompt}' ---")
    
    # Global Interaction: Ask for user approval before starting the whole process
    user_input = input(f"Do you approve starting the agent process for prompt: '{prompt}'? (y/n): ").strip().lower()
    if user_input != 'y':
        print("[SYSTEM] Agent process denied by user. Exiting.")
        return

    messages = [{"role": "user", "content": prompt}]
    
    while True:
        print("\n[Agent is thinking...]")
        # Call Ollama with the current conversation history and available tools
        response = ollama.chat(
            model=model_name,
            messages=messages,
            tools=tools_schema
        )
        
        message = response['message']
        messages.append(message)
        
        # If the model didn't call any tools, it means it has a final answer
        if not message.get('tool_calls'):
            print("\n=== Final Response ===")
            print(message.get('content'))
            break
            
        # If the model wants to call tools, process them
        for tool_call in message['tool_calls']:
            func_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            print(f"\n[SYSTEM] The model wants to run tool: '{func_name}'")
            print(f"[SYSTEM] With arguments: {json.dumps(args, indent=2)}")
            
            # 3. Global Interaction: Ask for user approval before each tool run
            user_input = input(f"Do you approve running the tool '{func_name}'? (y/n): ").strip().lower()
            
            if user_input == 'y':
                print("[SYSTEM] Tool call approved.")
                func = available_tools.get(func_name)
                
                if func:
                    # Execute the tool
                    try:
                        result = func(**args)
                        print(f"[SYSTEM] Tool result: {result}")
                        
                        # Append the tool result to the conversation history
                        messages.append({
                            "role": "tool",
                            "name": func_name,
                            "content": str(result)
                        })
                    except Exception as e:
                        print(f"[SYSTEM] Error executing tool: {e}")
                        messages.append({
                            "role": "tool",
                            "name": func_name,
                            "content": f"Error executing tool: {str(e)}"
                        })
                else:
                    print(f"[SYSTEM] Error: Tool '{func_name}' not found.")
                    messages.append({
                        "role": "tool",
                        "name": func_name,
                        "content": f"Error: Tool '{func_name}' not found."
                    })
            else:
                print("[SYSTEM] Tool call denied by user.")
                # Inform the model that the user denied the tool execution
                messages.append({
                    "role": "tool",
                    "name": func_name,
                    "content": "User denied the execution of this tool. Please provide an alternative response or ask the user for clarification."
                })

if __name__ == "__main__":
    # You can change 'llama3.1' to whatever model you have installed in Ollama that supports tools
    # e.g., 'llama3.2', 'mistral', 'qwen2.5'
    MODEL_NAME = "qwen3:1.7b" 
    
    USER_PROMPT = "What is the weather in Haifa and based on this recommend me an activity suitable for the weather."
    
    try:
        run_agent(USER_PROMPT, model_name=MODEL_NAME)
    except ollama.ResponseError as e:
        print(f"\n[ERROR] Ollama API Error: {e}")
        print(f"Make sure Ollama is running and you have pulled the '{MODEL_NAME}' model.")
        print(f"Run: ollama run {MODEL_NAME}")
