# Psychotherapy(Beta)

A Streamlit application that generates SOAP (Subjective, Objective, Assessment, Plan) notes and summaries from psychotherapy transcripts. The app allows users to upload text files containing psychotherapy session transcripts and get a formatted SOAP note along with summaries for both clinician and client.

## Features

- **File Upload**: Supports PDF, DOCX, and TXT file uploads.
- **SOAP Note Generation**: Automatically generates SOAP notes from the psychotherapy transcript.
- **Clinician & Client Summaries**: Provides summaries for both clinician and client.
- **Token Management**: Handles long transcripts by truncating text to fit within the model's token limit.

## Live Demo

You can try the live demo of the application at [Psycho-Therapy Scribing](https://psycho-therapy-scribing.streamlit.app/).

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Streamlit
- OpenAI Python library
- PyPDF2
- python-docx
- tiktoken

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/psychotherapy-soap-note-generator.git
