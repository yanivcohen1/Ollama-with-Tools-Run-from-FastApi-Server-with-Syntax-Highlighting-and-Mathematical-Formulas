import os
import ollama
import json
import inspect
from docstring_parser import parse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 1. Define the actual Python functions (Tools)
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    
    :param city: The city to get the weather for, e.g. Haifa
    """
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
        param_doc = next((p.description for p in doc.params if p.arg_name == param_name), "")
        
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

# Global conversation history to remember previous messages in the session
sessions = {}

async def agent_generator(session_id: str, prompt: str = None, action: str = "prompt", pending_tool_calls: list = None, remaining_tools: list = None):
    if session_id not in sessions:
        sessions[session_id] = {
            "chat_history": [
                {"role": "system", "content": "You are a helpful assistant. You must use the provided tools to answer the user's request whenever possible. Do not make up answers if a tool exists for it."}
            ],
            "auto_approve": False
        }
        
    session = sessions[session_id]
    chat_history = session["chat_history"]
    auto_approve = session["auto_approve"]
    
    if action == "prompt":
        # Reset auto_approve for every new user query
        session["auto_approve"] = False
        session["tools_denied_this_turn"] = False
        auto_approve = False
        
        # Clean up any dangling tool calls from previous interrupted turns
        cleaned_history = []
        for msg in chat_history:
            if msg["role"] == "tool":
                continue
            if msg["role"] == "assistant" and "tool_calls" in msg:
                if msg.get("content"):
                    cleaned_history.append({"role": "assistant", "content": msg["content"]})
                continue
            cleaned_history.append(msg)
        session["chat_history"] = cleaned_history
        chat_history = session["chat_history"]
        
        chat_history.append({"role": "user", "content": prompt})
        yield f"data: {json.dumps({'type': 'status', 'content': 'Agent is thinking...'})}\n\n"
    elif action == "auto_approve":
        session["auto_approve"] = True
        auto_approve = True
        action = "approve_tool"
        
    if action in ["approve_tool", "deny_tool"]:
        for tool_call in pending_tool_calls:
            func_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            if action == "approve_tool":
                yield f"data: {json.dumps({'type': 'status', 'content': f'Executing tool: {func_name}'})}\n\n"
                func = available_tools.get(func_name)
                if func:
                    try:
                        result = func(**args)
                        yield f"data: {json.dumps({'type': 'tool_result', 'content': f'Tool {func_name} result: {result}'})}\n\n"
                        chat_history.append({
                            "role": "tool",
                            "name": func_name,
                            "content": str(result)
                        })
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'status', 'content': f'Error executing tool: {str(e)}'})}\n\n"
                        chat_history.append({
                            "role": "tool",
                            "name": func_name,
                            "content": f"Error executing tool: {str(e)}"
                        })
                else:
                    yield f"data: {json.dumps({'type': 'status', 'content': f'Error: Tool {func_name} not found.'})}\n\n"
                    chat_history.append({
                        "role": "tool",
                        "name": func_name,
                        "content": f"Error: Tool '{func_name}' not found."
                    })
            else: # deny_tool
                yield f"data: {json.dumps({'type': 'status', 'content': f'Tool {func_name} denied by user.'})}\n\n"
                chat_history.append({
                    "role": "tool",
                    "name": func_name,
                    "content": "User denied the execution of this tool. Please respond to the user without using this tool."
                })
                remaining_tools = [] # Stop asking for remaining tools in this batch
                session["tools_denied_this_turn"] = True
                
        # If there are more tools to approve, ask for the next one immediately
        if remaining_tools and not auto_approve:
            next_tool = remaining_tools[0]
            yield f"data: {json.dumps({'type': 'tool_approval_request', 'tool_calls': [next_tool], 'remaining_tools': remaining_tools[1:]})}\n\n"
            return # End stream, wait for user decision on the next tool
            
        # If auto_approve is true, we need to execute the remaining tools automatically
        if remaining_tools and auto_approve:
            for tool_call in remaining_tools:
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                yield f"data: {json.dumps({'type': 'status', 'content': f'Auto-approving tool: {func_name}'})}\n\n"
                
                func = available_tools.get(func_name)
                if func:
                    try:
                        result = func(**args)
                        yield f"data: {json.dumps({'type': 'tool_result', 'content': f'Tool {func_name} result: {result}'})}\n\n"
                        chat_history.append({
                            "role": "tool",
                            "name": func_name,
                            "content": str(result)
                        })
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'status', 'content': f'Error executing tool: {str(e)}'})}\n\n"
                        chat_history.append({
                            "role": "tool",
                            "name": func_name,
                            "content": f"Error executing tool: {str(e)}"
                        })
                else:
                    yield f"data: {json.dumps({'type': 'status', 'content': f'Error: Tool {func_name} not found.'})}\n\n"
                    chat_history.append({
                        "role": "tool",
                        "name": func_name,
                        "content": f"Error: Tool '{func_name}' not found."
                    })
                    
        yield f"data: {json.dumps({'type': 'status', 'content': 'Agent is thinking...'})}\n\n"
    
    while True:
        print(f"--- Calling Ollama with chat_history: {json.dumps(chat_history, indent=2)} ---")
        # Call Ollama with the current conversation history and available tools
        response_stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=chat_history,
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
                full_content += msg['content']
                if is_final_response:
                    yield f"data: {json.dumps({'type': 'token', 'content': msg['content']})}\n\n"
                    await asyncio.sleep(0.01) # Small delay to allow UI to update smoothly
        
        if is_final_response:
            # Clean up previous tool calls and tool results to prevent the model from repeating them
            cleaned_history = []
            for msg in chat_history:
                if msg["role"] == "tool":
                    continue
                if msg["role"] == "assistant" and "tool_calls" in msg:
                    if msg.get("content"):
                        cleaned_history.append({"role": "assistant", "content": msg["content"]})
                    continue
                cleaned_history.append(msg)
            
            session["chat_history"] = cleaned_history
            session["chat_history"].append({"role": "assistant", "content": full_content})
            
            # Reset auto_approve at the end of the turn
            session["auto_approve"] = False
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            break
            
        # Convert tool_calls to dicts for JSON serialization
        serializable_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                serializable_tool_calls.append(tc)
            else:
                serializable_tool_calls.append({
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
                
        # If the model wants to call tools, process them
        chat_history.append({"role": "assistant", "content": full_content, "tool_calls": serializable_tool_calls})
        
        if not auto_approve:
            if serializable_tool_calls:
                if session.get("tools_denied_this_turn", False):
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Agent tried to call tools again, but was blocked because of previous denial.'})}\n\n"
                    
                    # Clean up history before ending the turn
                    cleaned_history = []
                    for msg in chat_history:
                        if msg["role"] == "tool":
                            continue
                        if msg["role"] == "assistant" and "tool_calls" in msg:
                            if msg.get("content"):
                                cleaned_history.append({"role": "assistant", "content": msg["content"]})
                            continue
                        cleaned_history.append(msg)
                    session["chat_history"] = cleaned_history
                    
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                    
                # If there are multiple tools, we only ask for the first one to allow step-by-step approval
                first_tool = serializable_tool_calls[0]
                yield f"data: {json.dumps({'type': 'tool_approval_request', 'tool_calls': [first_tool], 'remaining_tools': serializable_tool_calls[1:]})}\n\n"
                break # End stream, wait for user decision
            else:
                break
            
        # If auto_approve is True, execute tools and continue loop
        for tool_call in serializable_tool_calls:
            func_name = tool_call['function']['name']
            args = tool_call['function']['arguments']
            
            yield f"data: {json.dumps({'type': 'status', 'content': f'Auto-approving tool: {func_name}'})}\n\n"
            
            func = available_tools.get(func_name)
            
            if func:
                try:
                    result = func(**args)
                    yield f"data: {json.dumps({'type': 'tool_result', 'content': f'Tool {func_name} result: {result}'})}\n\n"
                    
                    chat_history.append({
                        "role": "tool",
                        "name": func_name,
                        "content": str(result)
                    })
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'status', 'content': f'Error executing tool: {str(e)}'})}\n\n"
                    chat_history.append({
                        "role": "tool",
                        "name": func_name,
                        "content": f"Error executing tool: {str(e)}"
                    })
            else:
                yield f"data: {json.dumps({'type': 'status', 'content': f'Error: Tool {func_name} not found.'})}\n\n"
                chat_history.append({
                    "role": "tool",
                    "name": func_name,
                    "content": f"Error: Tool '{func_name}' not found."
                })

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "default")
    prompt = data.get("prompt", "")
    action = data.get("action", "prompt")
    tool_calls = data.get("tool_calls", [])
    remaining_tools = data.get("remaining_tools", [])
    
    # If there are remaining tools after an approval/denial, we need to process them next
    if remaining_tools and action in ["approve_tool", "deny_tool"]:
        # We will handle the current tool, then immediately ask for the next one
        pass
        
    return StreamingResponse(agent_generator(session_id, prompt, action=action, pending_tool_calls=tool_calls, remaining_tools=remaining_tools), media_type="text/event-stream")

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

if __name__ == "__main__":
    print("Starting web server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)