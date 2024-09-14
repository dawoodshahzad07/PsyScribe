import streamlit as st
from openai import OpenAI
import PyPDF2
import docx
import tiktoken  # For token counting

# Set up the OpenAI client for Upstage API
client = OpenAI(
    api_key="up_QnlPsfFqCqUDAfi3N68kMRgzDGjix",  # Replace with your OpenAI API key
    base_url="https://api.upstage.ai/v1/solar"
)

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

# Helper function to extract text from uploaded file
def extract_text_from_file(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    elif file.type == "text/plain":
        return file.read().decode('utf-8')
    else:
        return None

# Streamlit UI
st.title("Generate SOAP Notes and Summaries from Psychotherapy Transcript")

# File uploader
uploaded_file = st.file_uploader("Upload a psychotherapy transcript (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Extract text from the uploaded document
    transcript_text = extract_text_from_file(uploaded_file)
    
    if transcript_text:
        st.write("Transcript uploaded successfully.")
        
        if st.button("Generate SOAP Note and Summaries"):
            st.write("Generating SOAP note and summaries...")
            
            # Stream and display the response
            response_container = st.empty()  # Placeholder for streamed response
            full_response = ""

            # Get the response in chunks and update the placeholder
            for chunk in get_streamed_response(transcript_text):
                full_response += chunk
                response_container.markdown(full_response.replace('\n', '  \n'))  # Ensure line breaks are displayed
    else:
        st.error("Unsupported file format or could not extract text from the file.")
