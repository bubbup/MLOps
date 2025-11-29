import gradio as gr
import requests
import os
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

print("STARTING APP INITIALIZATION")

# Ollama Local Server Configuration
LOCAL_OLLAMA_ENDPOINT = "http://localhost:11435/api/generate"
print(f"Ollama endpoint set to: {LOCAL_OLLAMA_ENDPOINT}")

def generate_response(prompt: str, model_name: str, temperature: float = 0.7):
    """Generate a response using the selected Ollama model."""
    if not prompt:
        return "Please enter a prompt or question."
    
    try:
        # Correct format for /api/generate endpoint
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature
        }
        
        response = requests.post(LOCAL_OLLAMA_ENDPOINT, json=data, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        content = result.get("response", "No response generated.")
        return content
    
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model may be processing a complex query. Try again or use a simpler prompt."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running (ollama serve in another terminal)."
    except Exception as e:
        return f"Error: {str(e)}"

print("\nCreating Gradio interface...")

# Model selector dropdown
model_dropdown = gr.Dropdown(
    choices=["deepseek-r1:latest"],
    value="deepseek-r1:latest",
    label="Select Model",
    info="Deepseek R1 - Local reasoning model"
)

# Temperature slider for controlling creativity
temperature_slider = gr.Slider(
    minimum=0.0,
    maximum=1.0,
    value=0.7,
    step=0.1,
    label="Temperature (Creativity)",
    info="0.0 = Deterministic, 1.0 = Creative"
)

# Create the Gradio Interface
gui = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(
            lines=5,
            label="Your Prompt / Question",
            placeholder="Ask me anything or give me a task...",
            info="Be specific for better results"
        ),
        model_dropdown,
        temperature_slider
    ],
    outputs=gr.Textbox(
        label="AI Response",
        lines=8,
        interactive=False
    ),
    title=" Local AI Assistant",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

print("Gradio interface created successfully")

# FASTAPI APP - For serving the Gradio interface
print("Creating FastAPI app...")

app = FastAPI(
    title="Local AI Assistant API",
    description="A local AI assistant powered by Deepseek R1",
    version="1.0.0"
)

print("Mounting Gradio app to FastAPI...")

# Mount the Gradio interface to FastAPI
app = gr.mount_gradio_app(app, gui, path="/")

print("APP INITIALIZED SUCCESSFULLY!")
print("Server is ready. Access it at: http://127.0.0.1:8000")