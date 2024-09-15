import streamlit as st
from openai import OpenAI
import PyPDF2
import docx
import tiktoken  # For token counting
from io import BytesIO

# Set up the OpenAI client
client = OpenAI(
    api_key="up_QnlPsfFqCqUDAfi3N68kMRgzDGjix",  # Replace with your OpenAI API key
    base_url="https://api.upstage.ai/v1/solar"
)

# List of psychotherapy-related keywords for context filtering
psychotherapy_keywords = ["therapy", "therapist", "session", "client", "counseling", "psychotherapy", "mental health"]

# Function to count tokens (requires tiktoken library)
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")  # Choose appropriate encoding
    tokens = encoding.encode(text)
    return len(tokens)

# Function to truncate text to a specified number of tokens
def truncate_text(prompt, text, max_tokens):
    encoding = tiktoken.get_encoding("cl100k_base")
    prompt_tokens = len(encoding.encode(prompt))
    content_tokens = max_tokens - prompt_tokens
    tokens = encoding.encode(text)
    
    if len(tokens) > content_tokens:
        tokens = tokens[:content_tokens]
        text = encoding.decode(tokens)
    
    return text

# Function to handle streaming response from Upstage's solar-pro model
def get_streamed_response(text):
    # Token limit for the model
    max_tokens = 3000

    # Define the prompt
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
    
    # Truncate the text to fit within the token limit, including the prompt
    truncated_text = truncate_text(prompt, text, max_tokens)

    # Modify the prompt to specifically request SOAP note format and summaries
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
    
    # Stream the API response as the output generates
    stream = client.chat.completions.create(
        model="solar-pro",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        stream=True,
    )

    # Yield chunks of streamed content
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Helper function to extract text from file-like objects (hardcoded or uploaded)
def extract_text_from_file(file, file_type=None):
    if file_type is None:  # Handling BytesIO objects (hardcoded files)
        content = file.read()
        if content.startswith(b'%PDF-'):
            reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif content.startswith(b'PK\x03\x04'):
            doc = docx.Document(BytesIO(content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        else:
            return content.decode('utf-8')
    else:  # Handling Streamlit file uploader objects
        if file_type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        elif file_type == "text/plain":
            return file.read().decode('utf-8')
        else:
            return None

# Function to check if the transcript is related to psychotherapy
def is_relevant_psychotherapy_text(text):
    # Check if any psychotherapy keywords are in the text
    return any(keyword.lower() in text.lower() for keyword in psychotherapy_keywords)

# Hardcoded file path (example transcript)
hardcoded_file_path = "psychotherapy_transcript.txt"  # Replace with the actual file path

# Streamlit UI
st.title("Generate SOAP Notes and Summaries from Psychotherapy Transcript")

# 1. Button to upload and process a hardcoded example file
if st.button("Upload Example Dataset"):
    # Load the file from the hardcoded path
    try:
        with open(hardcoded_file_path, "rb") as file:
            # Directly read file content
            file_content = file.read()
            transcript_text = extract_text_from_file(BytesIO(file_content))

        if transcript_text:
            st.write("Transcript loaded successfully.")
            
            # Check if the transcript is related to psychotherapy
            if is_relevant_psychotherapy_text(transcript_text):
                st.write("Generating SOAP note and summaries...")
                
                # Stream and display the response
                response_container = st.empty()  # Placeholder for streamed response
                full_response = ""

                # Get the response in chunks and update the placeholder
                for chunk in get_streamed_response(transcript_text):
                    full_response += chunk
                    response_container.markdown(full_response.replace('\n', '  \n'))  # Ensure line breaks are displayed
            else:
                st.error("The transcript does not appear to be related to psychotherapy.")
    except Exception as e:
        st.error(f"Failed to load example dataset. Error: {e}")

# 2. File uploader for users to upload their own file
uploaded_file = st.file_uploader("Upload your own psychotherapy transcript (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Process the uploaded file if it exists
if uploaded_file is not None:
    transcript_text = extract_text_from_file(uploaded_file, uploaded_file.type)

    if transcript_text:
        st.write("Transcript loaded successfully.")
        
        # Check if the transcript is related to psychotherapy
        if is_relevant_psychotherapy_text(transcript_text):
            st.write("Generating SOAP note and summaries...")
            
            # Stream and display the response
            response_container = st.empty()  # Placeholder for streamed response
            full_response = ""

            # Get the response in chunks and update the placeholder
            for chunk in get_streamed_response(transcript_text):
                full_response += chunk
                response_container.markdown(full_response.replace('\n', '  \n'))  # Ensure line breaks are displayed
        else:
            st.error("The transcript does not appear to be related to psychotherapy.")
else:
    st.write("No file uploaded. You can upload a file using the file uploader above.")
