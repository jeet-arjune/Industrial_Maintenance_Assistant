
# AI-Powered Industrial Maintenance Assistant

Maintena AI is an educational chatbot that helps users explore industrial maintenance information from technical documents.

## Features
- Ask industrial maintenance questions
- Upload PDF and TXT manuals
- Search relevant manual sections
- Generate answers using a Hugging Face language model
- View source sections for answers
- Browse the maintenance guide

## Technologies
- Python
- Streamlit
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- pypdf

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Project files
- `app.py` — main application
- `maintenance_manual.txt` — default maintenance guide
- `requirements.txt` — required packages
- `README.md` — project documentation

## Safety note
This educational prototype does not replace manufacturer instructions, approved site procedures, or qualified maintenance personnel.
