import gradio as gr
import torch
from transformers import pipeline

# Speech-to-Text (Whisper)
stt = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Text Generation (Mistral/GPT)
chat = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

def assistant(audio):
    # Convert speech to text
    text = stt(audio)["text"]

    # Generate AI response
    response = chat(text, max_length=100)[0]["generated_text"]

    return response

demo = gr.Interface(
    fn=assistant,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs=gr.Textbox()
)

demo.launch()
