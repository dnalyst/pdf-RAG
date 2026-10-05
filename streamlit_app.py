# import streamlit as st

# # -----------------------------
# # Page Configuration
# # -----------------------------
# st.set_page_config(
#     page_title="PDF RAG Assistant",
#     page_icon="📄",
#     layout="wide"
# )

# # -----------------------------
# # Sidebar
# # -----------------------------
# with st.sidebar:
#     st.title("📄 PDF RAG Assistant")
#     st.markdown("---")
#     st.write("Upload a PDF and ask questions about it.")
#     st.button("🗑️ Clear Chat")

# # -----------------------------
# # Main Page
# # -----------------------------
# st.title("🤖 PDF RAG Assistant")

# st.write("Ask questions about your PDF using Google Gemini and FAISS.")

# uploaded_file = st.file_uploader(
#     "Upload your PDF",
#     type=["pdf"]
# )

# question = st.text_input(
#     "Ask a question"
# )

# if st.button("Ask"):
#     if not uploaded_file:
#         st.warning("Please upload a PDF first.")
#     elif not question:
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Generating answer..."):
#             st.success("UI is working! Backend will be connected next.")

import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS





EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_MODEL = "gemini-3.5-flash"

@st.cache_resource
def initialize_gemini():

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found."
        )

    return genai.Client(api_key=api_key)

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

def build_vector_store(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = load_embedding_model()

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    

    return vector_db


def ask_question(client, vector_db, question):

    docs = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in docs
    )
    

    prompt = f"""
You are an AI assistant.

Answer ONLY using the context below.

Cite the source page number whenever you provide factual information.
Use citations in this format: [Page 12].
Only cite page numbers that appear in the provided context.
Do not invent page numbers.

If the answer is not present in the context, reply:

"I could not find the answer in the provided document."
Context:
{context}

Question:
{question}
"""

    try:
        response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

        return response.text

    except Exception as e:
        return f"Gemini is temporarily unavailable. Please try again in a moment.\n\nError: {e}"
# ==========================================================
# Streamlit UI
# ==========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    text-align: center;
    color: #2563EB;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    color: white;

}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: white !important;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

section[data-testid="stSidebar"] .stButton > button {
    background-color: #2563EB;
    color: white !important;
    border: none;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #1D4ED8;
    color: white !important;
}

.stTextInput input {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
st.title("📄 PDF AI Assistant")

st.caption("Chat with your PDF using Retrieval-Augmented Generation (RAG).")




# Sidebar
with st.sidebar:

    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if st.button("📂 Load PDF"):
        st.session_state["load_pdf"] = True

    st.divider()

    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### ⚡ Powered by")

    st.markdown("""
    - 🤖 Google Gemini
    - 🦜 LangChain
    - 📚 FAISS
    """)
# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []


client = initialize_gemini()

embeddings = load_embedding_model()

# Load Vector Store
if st.session_state.get("load_pdf"):

    if uploaded_file is None:

        st.warning("Please upload a PDF first.")

    else:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            temp_pdf_path = temp_pdf.name

        try:
           with st.spinner("Processing PDF..."):
              vector_db = build_vector_store(temp_pdf_path)

              st.session_state["vector_db"] = vector_db
              st.session_state["loaded_file_name"] = uploaded_file.name

              st.success(f"PDF Loaded Successfully: {uploaded_file.name}")

        except Exception as e:
              st.error(f"Could not process this PDF: {e}")

        finally:
              if os.path.exists(temp_pdf_path):
                 os.remove(temp_pdf_path)

        st.session_state["load_pdf"] = False
# Question
st.subheader("💬 Chat")

question = st.text_input(
    "Ask a Question",
    placeholder="Type your question here..."
)

if st.button("🤖 Ask AI"):

    if "vector_db" not in st.session_state:

        st.warning("Please load a PDF first.")

    elif not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            answer = ask_question(
                client,
                st.session_state["vector_db"],
                question
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])