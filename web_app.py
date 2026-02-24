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
chat_history = [
    {"role": "system", "content": "You are a helpful assistant. You must use the provided tools to answer the user's request whenever possible. Do not make up answers if a tool exists for it."}
]

async def agent_generator(prompt: str, model_name: str = "qwen3:1.7b"):
    global chat_history
    chat_history.append({"role": "user", "content": prompt})
    
    yield f"data: {json.dumps({'type': 'status', 'content': 'Agent is thinking...'})}\n\n"
    
    while True:
        # Call Ollama with the current conversation history and available tools
        response_stream = ollama.chat(
            model=model_name,
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
            chat_history.append({"role": "assistant", "content": full_content})
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            break
            
        # If the model wants to call tools, process them
        chat_history.append({"role": "assistant", "content": full_content, "tool_calls": tool_calls})
        
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
            else:
                func_name = tool_call.function.name
                args = tool_call.function.arguments
            
            yield f"data: {json.dumps({'type': 'status', 'content': f'The model wants to run tool: {func_name} with args: {json.dumps(args)}'})}\n\n"
            
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
    prompt = data.get("prompt", "")
    return StreamingResponse(agent_generator(prompt), media_type="text/event-stream")

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

if __name__ == "__main__":
    print("Starting web server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)