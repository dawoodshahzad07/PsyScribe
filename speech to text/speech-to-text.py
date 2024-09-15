import streamlit as st
import os
import dotenv
import assemblyai as aai
import requests

# Load environment variables
dotenv.load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")  # Use environment variable for AssemblyAI key

st.set_page_config(page_title="Speech to Text", page_icon="🎙️")

st.markdown("<h1 style='text-align: center;'>Speech to Text 🎙️</h1>", unsafe_allow_html=True)

# Form to upload the audio file
with st.form("my_form"):
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    submit_button = st.form_submit_button(label="Submit")

st.markdown("<h3 style='text-align: center;'>Try it out with any audio file!</h3>", unsafe_allow_html=True)

# Flag to indicate the transcription process status
is_processing = False

if submit_button and audio_file is None:
    st.warning("Please upload an audio file.")

elif submit_button and audio_file is not None:
    is_processing = True  # Set flag to True when processing starts

    # Display processing message
    if is_processing:
        st.info("Processing your audio file...")

    # Read the contents of the file (in-memory)
    audio_data = audio_file.read()

    # Upload the file in chunks to avoid timeout
    try:
        headers = {
            'authorization': aai.settings.api_key,
            'content-type': 'application/json'
        }
        
        # Step 1: Initialize upload
        upload_url = "https://api.assemblyai.com/v2/upload"
        response = requests.post(upload_url, headers=headers, data=audio_data)

        if response.status_code != 200:
            st.error("Error in uploading the audio file.")
            is_processing = False
        else:
            upload_url = response.json()['upload_url']

            # Step 2: Submit the uploaded file for transcription
            transcriber = aai.Transcriber()
            transcript_response = transcriber.transcribe(upload_url)

            # Check for errors in transcription
            if transcript_response.status == aai.TranscriptStatus.error:
                st.error(f"Error in transcription: {transcript_response.error}")
            else:
                st.success("Transcription completed successfully!")
                st.write(transcript_response.text)
                is_processing = False  # Set flag to False when transcription is done

    except requests.exceptions.Timeout:
        st.error("The upload operation timed out. Please try again with a smaller file or better network connection.")
        is_processing = False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        is_processing = False  # Set flag to False if an error occurs
