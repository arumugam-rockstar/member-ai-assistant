import os
import zipfile
import shutil
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

EXTRACT_FOLDER = "./extracted_docs"
DB_DIR = "./chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def find_zip_file():
    """Finds knowledge base zip file regardless of capitalization."""
    for filename in os.listdir("."):
        if filename.lower() == "knowledge_base.zip":
            return filename
    return None

def build_or_load_vector_db():
    zip_filename = find_zip_file()

    # If zip file exists, extract it and FORCE rebuild ChromaDB to purge stale data
    if zip_filename:
        logger.info(f"--> Found {zip_filename}. Extracting and rebuilding Vector DB...")

        # Clean old extracted docs & old vector database
        if os.path.exists(EXTRACT_FOLDER):
            shutil.rmtree(EXTRACT_FOLDER)
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)

        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)

        logger.info("--> Loading extracted Markdown documents into memory...")
        loader = DirectoryLoader(
            EXTRACT_FOLDER, 
            glob="**/*.md",  # Exclusively load Markdown files
            loader_cls=TextLoader, 
            loader_kwargs={'autodetect_encoding': True}
        )
        raw_documents = loader.load()

        logger.info(f"--> Splitting {len(raw_documents)} document(s) into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_documents(raw_documents)

        logger.info(f"--> Building fresh Vector DB with {len(chunks)} chunks...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_DIR
        )
        return vector_db

    # Fallback to existing disk DB if zip is not present
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        logger.info("--> Loading existing Vector Database from disk...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

    raise FileNotFoundError("knowledge_base.zip was not found in the repository!")

vector_store = build_or_load_vector_db()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "PMICC Member Assistant API"
    }), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        # Retrieve Top 5 relevant chunks
        relevant_docs = vector_store.similarity_search(user_message, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        system_prompt = f"""You are the official assistant for the PMI Chennai Chapter.
Answer the user's question accurately using ONLY the retrieved facts from our knowledge base below.
If the information is not present in the context, politely state that you do not have that specific information.

=== RETRIEVED KNOWLEDGE BASE SNIPPETS ===
{context_text}
==========================================
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = client.chat_completion(
            messages,
            max_tokens=300,
            temperature=0.1
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply}), 200

    except Exception as e:
        logger.error(f"Error handling chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
