import os
import ollama
import json
import inspect
from docstring_parser import parse
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

# 1. Define the actual Python functions (Tools)
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    
    :param city: The city to get the weather for, e.g. Haifa
    """
    print(f"    [Executing get_weather for {city}]")
    # In a real app, you would call a weather API here
    weather_data = {
        "haifa": "sunny and 25°C",
        "tel aviv": "cloudy and 20°C",
        "jerusalem": "windy and 18°C"
    }
    return weather_data.get(city.lower(), "unknown weather")

def recommend_activity(weather: str) -> str:
    """
    Recommend an activity based on the weather.
    
    :param weather: The weather condition, e.g. sunny and 25°C
    """
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

# 2. Automatically build the tool schemas for Ollama using reflection
def generate_tool_schema(func):
    sig = inspect.signature(func)
    doc = parse(func.__doc__)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # Find the parameter description from the docstring
        param_doc = next((p.description for p in doc.params if p.arg_name == param_name), "")
        
        # Map Python types to JSON schema types
        param_type = "string" # Default
        if param.annotation == int:
            param_type = "integer"
        elif param.annotation == float:
            param_type = "number"
        elif param.annotation == bool:
            param_type = "boolean"
            
        properties[param_name] = {
            "type": param_type,
            "description": param_doc
        }
        
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
            
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.short_description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

tools_schema = [generate_tool_schema(func) for func in available_tools.values()]

def run_agent(prompt: str, model_name: str = "qwen3:1.7b"):
    print(f"\n--- Starting Agent with prompt: '{prompt}' ---")
    
    user_promp = input(f"Do you want to use the prompt '{prompt}'? (y/n): ").strip().lower()
    if user_promp != 'y':
        prompt = input("Please enter your desired prompt: ").strip()
    
    # Global Interaction: Ask for user approval before starting the whole process
    user_input = input(f"Do you approve starting the agent process for prompt with auto approval? (y/n/c)\n(y - run with auto approval, n - user needs to approve every step, c - cancel all the process): ").strip().lower()
    
    if user_input == 'c':
        print("[SYSTEM] Agent process canceled by user. Exiting.")
        return
    
    auto_approve = (user_input == 'y')

    messages = [
        {"role": "system", "content": "You are a helpful assistant. You must use the provided tools to answer the user's request whenever possible. Do not make up answers if a tool exists for it."},
        {"role": "user", "content": prompt}
    ]
    
    while True:
        print("\n[Agent is thinking...]")
        # Call Ollama with the current conversation history and available tools
        response_stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools_schema,
            stream=True
        )
        
        full_content = ""
        tool_calls = []
        
        is_final_response = True
        
        for chunk in response_stream:
            msg = chunk['message']
            
            if msg.get('tool_calls'):
                is_final_response = False
                tool_calls.extend(msg['tool_calls'])
                
            if msg.get('content'):
                if is_final_response and not full_content:
                    print("\n=== Final Response ===")
                full_content += msg['content']
                if is_final_response:
                    print(msg['content'], end='', flush=True)
        
        if is_final_response:
            print() # Print a newline at the end of the stream
            messages.append({"role": "assistant", "content": full_content})
            break
            
        # If the model wants to call tools, process them
        messages.append({"role": "assistant", "content": full_content, "tool_calls": tool_calls})
        
        for tool_call in tool_calls:
            # Handle both dict and object representations of tool_calls
            if isinstance(tool_call, dict):
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
            else:
                func_name = tool_call.function.name
                args = tool_call.function.arguments
            
            print(f"\n[SYSTEM] The model wants to run tool: '{func_name}'")
            print(f"[SYSTEM] With arguments: {json.dumps(args, indent=2)}")
            
            # 3. Global Interaction: Ask for user approval before each tool run if not auto-approved
            if auto_approve:
                print("[SYSTEM] Auto-approving tool call.")
                tool_approved = True
            else:
                tool_input = input(f"Do you approve running the tool '{func_name}'? (y/n): ").strip().lower()
                tool_approved = (tool_input == 'y')
            
            if tool_approved:
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
