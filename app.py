import os
import zipfile
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

# LangChain & Vector DB imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enables frontend requests without CORS issues

# Fetch Hugging Face API token securely from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

ZIP_PATH = "knowledge_base.zip"
EXTRACT_FOLDER = "./extracted_docs"
DB_DIR = "./chroma_db"

# 1. Initialize Free Local Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_or_load_vector_db():
    """Extracts zip archive, chunks documents, and creates/loads ChromaDB."""
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        logger.info("Loading existing Vector Database from disk...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

    logger.info("Extracting knowledge base zip...")
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
    else:
        logger.warning(f"{ZIP_PATH} not found. Make sure knowledge_base.zip is in your repository root.")

    logger.info("Loading documents into memory...")
    # Recursively loads all files inside extracted folder
    loader = DirectoryLoader(
        EXTRACT_FOLDER, 
        glob="**/*.*", 
        loader_cls=TextLoader, 
        loader_kwargs={'autodetect_encoding': True}
    )
    raw_documents = loader.load()

    logger.info(f"Splitting {len(raw_documents)} raw document(s) into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_documents)

    logger.info(f"Generating Vector Database with {len(chunks)} text chunks...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    logger.info("Vector Database built successfully!")
    return vector_db

# Load/Build database on server start
vector_store = build_or_load_vector_db()

# --- Root Route (Health Check) ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "PMICC Member Assistant API",
        "message": "Backend service is up and running!"
    }), 200

# --- Chat Endpoint (RAG Pipeline) ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        # 2. Retrieve Top 3 Most Relevant Document Chunks
        relevant_docs = vector_store.similarity_search(user_message, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # 3. Formulate Prompt
        system_prompt = f"""You are the official assistant for our non-profit organization.
Answer the user's question accurately using ONLY the retrieved facts from our knowledge base below. 
If the information is not present in the context, reply politely stating that you do not have that specific information.

=== RETRIEVED KNOWLEDGE BASE SNIPPETS ===
{context_text}
==========================================
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # 4. Generate Answer via Hugging Face Client
        response = client.chat_completion(
            messages,
            max_tokens=300,
            temperature=0.1
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply}), 200

    except Exception as e:
        logger.error(f"Error handling chat request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
