# PDF RAG Assistant

A deployed Retrieval Augmented Generation application that lets users upload a PDF and ask questions about its contents. The system retrieves relevant passages using semantic similarity search and uses Google Gemini to generate answers grounded in the uploaded document.

**Live Demo:** https://pdf-rag-csuozjoreqomy7jic7pxc4.streamlit.app/

**GitHub:** https://github.com/dnalyst/pdf-RAG

## Features

* Upload any PDF directly through the web application
* Extract and split document text into overlapping chunks
* Generate embeddings using Sentence Transformers
* Store document embeddings in a FAISS vector database
* Retrieve the most relevant document chunks for each question
* Generate answers using Google Gemini
* Provide page level citations for retrieved information
* Avoid answering when the required information is not present in the document
* Deploy the application using Streamlit Community Cloud

## Tech Stack

* Python
* Streamlit
* Google Gemini API
* LangChain
* Hugging Face Sentence Transformers
* FAISS
* PyPDF
* python dotenv

## RAG Pipeline

```text
User uploads PDF
       |
       v
PDF text extraction
       |
       v
Text chunking
       |
       v
Sentence Transformer embeddings
       |
       v
FAISS vector database
       |
       v
User question
       |
       v
Similarity search
       |
       v
Top relevant document chunks
       |
       v
Prompt with retrieved context
       |
       v
Google Gemini
       |
       v
Grounded answer with page citation
```

## How It Works

1. The user uploads a PDF through the Streamlit interface.
2. The PDF is processed using PyPDF.
3. The extracted text is divided into overlapping chunks.
4. Each chunk is converted into an embedding using Sentence Transformers.
5. The embeddings are stored in a FAISS vector database.
6. When the user asks a question, the system performs similarity search to retrieve the most relevant chunks.
7. The retrieved context and question are sent to Google Gemini.
8. Gemini generates an answer using only the retrieved document context.
9. Page numbers from the source document are included in the response.
10. If the required information is not present in the retrieved context, the system returns a fallback response instead of inventing an answer.

## Local Setup

### Clone the repository

```bash
git clone https://github.com/dnalyst/pdf-RAG.git
cd pdf-RAG
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure the Gemini API key

Create a `.env` file in the project directory:

```text
GOOGLE_API_KEY=YOUR_API_KEY
```

Do not commit the `.env` file to GitHub.

### Run the application

```bash
streamlit run streamlit_app.py
```

## Project Structure

```text
pdf-RAG/
│
├── data/
├── app.py
├── app_learning.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## Key Design Decisions

### Document grounded generation

The application instructs Gemini to answer only from the retrieved document context. This reduces unsupported answers and keeps responses tied to the uploaded source.

### Page level citations

Each retrieved chunk retains its PDF page metadata. The page information is passed to the language model so the generated answer can reference the relevant source pages.

### Temporary document processing

Uploaded PDFs are written to a temporary location for processing rather than being permanently stored in the repository.

### Local vector search

FAISS provides efficient similarity search over the document embeddings without requiring an external vector database.

## Limitations

* Each uploaded PDF is processed when loaded into the application.
* The current application is designed for document level question answering rather than a persistent multi document knowledge base.
* Retrieval quality depends on document structure, chunking and embedding quality.
* The application currently retrieves a small number of relevant chunks for each question.

## Future Improvements

* Multi document collections
* Conversation aware retrieval
* Hybrid keyword and semantic search
* Improved retrieval evaluation
* Reranking of retrieved passages
* Persistent vector storage
* Metadata based filtering
* Better handling of tables and complex PDF layouts

## What This Project Demonstrates

This project demonstrates an end to end RAG workflow covering:

* Document ingestion
* Text preprocessing
* Chunking
* Embedding generation
* Vector similarity search
* Context construction
* Prompt design
* LLM based answer generation
* Source citation
* Hallucination control
* Web application development
* Cloud deployment