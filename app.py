import time
import random
import streamlit as st
import pyaudio
import wave
import numpy as np
import os
import requests
import dotenv
import json
from openai import OpenAI
import tiktoken
from io import BytesIO
import PyPDF2
import docx

# Load environment variables
dotenv.load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
client = OpenAI(
    api_key="up_QnlPsfFqCqUDAfi3N68kMRgzDGjix",  # Replace with your OpenAI API key
    base_url="https://api.upstage.ai/v1/solar"
)

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"
VOLUME_BOOST = 1.5

# Function to record and save audio with a timer
def record_audio(duration_seconds):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    st.write("Recording...")

    # Create a placeholder for the timer
    timer_placeholder = st.empty()
    start_time = time.time()

    for _ in range(0, int(RATE / CHUNK * duration_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
        elapsed_time = int(time.time() - start_time)
        timer_placeholder.text(f"Recording Time: {elapsed_time} seconds")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = np.clip(audio_data * VOLUME_BOOST, -32768, 32767).astype(np.int16)

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())

    st.write(f"Recording saved as {WAVE_OUTPUT_FILENAME}")

# Function to format transcription text
def format_transcription(transcript):
    formatted_text = "BEGIN TRANSCRIPT:\n\n"
    formatted_lines = []
    speakers = {'0': 'Speaker 1', '1': 'Speaker 2', '2': 'Speaker 3'}
    for utterance in transcript.get('utterances', []):
        speaker_id = utterance.get('speaker', '0')
        speaker = speakers.get(speaker_id, f"Speaker {random.randint(1, 100)}")
        text = utterance['text'].strip()
        if text:
            formatted_lines.append(f"{speaker}: {text}")
    formatted_text += '\n'.join(formatted_lines).replace('.', '.\n')
    return formatted_text

# Function to get streamed response from OpenAI
def get_streamed_response(text):
    max_tokens = 3000
    prompt = f"""
    Generate a detailed SOAP note and summaries based on the following psychotherapy transcript:

    {text}

    Format the response in this structure:
    Subjective:
    Objective:
    Assessment:
    Plan:

    Clinician Summary:
    Client Summary:
    """
    truncated_text = truncate_text(prompt, text, max_tokens)
    final_prompt = f"""
    Generate a detailed SOAP note and summaries based on the following psychotherapy transcript:

    {truncated_text}

    Format the response in this structure:
    Subjective:
    Objective:
    Assessment:
    Plan:

    Clinician Summary:
    Client Summary:
    """
    stream = client.chat.completions.create(
        model="solar-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_prompt}
        ],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Function to count tokens
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

# Function to truncate text to fit within token limits
def truncate_text(prompt, text, max_tokens):
    encoding = tiktoken.get_encoding("cl100k_base")
    prompt_tokens = len(encoding.encode(prompt))
    content_tokens = max_tokens - prompt_tokens
    tokens = encoding.encode(text)
    if len(tokens) > content_tokens:
        tokens = tokens[:content_tokens]
        text = encoding.decode(tokens)
    return text

# Streamlit app UI
st.title("Audio to SOAP Notes and Summaries")

# Input for recording duration in minutes
minutes = st.number_input("Enter recording duration (minutes):", min_value=1, max_value=60, value=1)
duration_seconds = minutes * 60

# Record button
if st.button("Start Recording"):
    record_audio(duration_seconds)

# Process recorded file
if os.path.exists(WAVE_OUTPUT_FILENAME):
    st.write(f"Processing file: {WAVE_OUTPUT_FILENAME}")

    with open(WAVE_OUTPUT_FILENAME, "rb") as audio_file:
        headers = {'authorization': ASSEMBLYAI_API_KEY, 'content-type': 'application/json'}
        upload_url = "https://api.assemblyai.com/v2/upload"
        response = requests.post(upload_url, headers=headers, data=audio_file.read())
        if response.status_code == 200:
            upload_url = response.json()['upload_url']
            transcript_request = {'audio_url': upload_url, 'speaker_labels': True}
            response = requests.post('https://api.assemblyai.com/v2/transcript', headers=headers, json=transcript_request)
            if response.status_code == 200:
                transcript_id = response.json()['id']
                while True:
                    result_response = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
                    result = result_response.json()
                    if result['status'] in ['completed', 'failed']:
                        break
                    time.sleep(10)
                if result['status'] == 'completed':
                    transcript = result
                    formatted_text = format_transcription(transcript)
                    file_path = "transcription.txt"
                    with open(file_path, "w") as file:
                        file.write(formatted_text)
                    st.success(f"Transcription saved to {file_path}. Generating SOAP notes...")
                    response_container = st.empty()
                    full_response = ""
                    for chunk in get_streamed_response(formatted_text):
                        full_response += chunk
                        response_container.markdown(full_response.replace('\n', '  \n'))
                else:
                    st.error(f"Error in transcription: {result['error']}")
            else:
                st.error("Error in submitting the transcription request.")
        else:
            st.error("Error in uploading the audio file.")
