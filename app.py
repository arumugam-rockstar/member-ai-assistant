import os
import zipfile
import shutil
import logging
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Load Hugging Face API Token
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

# Storage & Directory Definitions
EXTRACT_FOLDER = "./extracted_docs"
DB_DIR = "./chroma_db"

# Global Objects & Thread Lock for Safe Lazy Loading
embedding_model = None
vector_store = None
db_lock = threading.Lock()


def find_zip_file():
    """Case-insensitive scanner to find knowledge_base.zip in root dir."""
    for filename in os.listdir("."):
        if filename.lower() == "knowledge_base.zip":
            return filename
    return None


def get_vector_db():
    """Thread-safe lazy initialization for Chroma Vector DB.
    Uses FastEmbed (ONNX) to keep memory footprint under 150MB.
    """
    global vector_store, embedding_model

    with db_lock:
        if vector_store is not None:
            return vector_store

        # Initialize ONNX-based lightweight embedding model
        if embedding_model is None:
            logger.info("--> Initializing lightweight FastEmbed model (ONNX Runtime)...")
            embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        zip_filename = find_zip_file()

        # If vector storage folder doesn't exist yet, extract zip & build database
        if zip_filename and not os.path.exists(DB_DIR):
            logger.info(f"--> Found zip: '{zip_filename}'. Extracting and building Vector DB...")

            if os.path.exists(EXTRACT_FOLDER):
                shutil.rmtree(EXTRACT_FOLDER)

            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_FOLDER)

            logger.info("--> Loading Markdown documents...")
            loader = DirectoryLoader(
                EXTRACT_FOLDER, 
                glob="**/*.md", 
                loader_cls=TextLoader, 
                loader_kwargs={'autodetect_encoding': True}
            )
            raw_documents = loader.load()

            # Split raw text into manageable context chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = text_splitter.split_documents(raw_documents)

            logger.info(f"--> Persisting {len(chunks)} chunks into Chroma DB...")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=DB_DIR
            )
            return vector_store

        # Load existing vector database from disk if present
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            logger.info("--> Loading existing Vector Database from disk...")
            vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
            return vector_store

        raise FileNotFoundError("knowledge_base.zip not found and no Chroma DB exists!")


@app.route("/", methods=["GET"])
def home():
    """Healthcheck endpoint for Render."""
    return jsonify({
        "status": "online",
        "service": "PMICC AI Assistant API",
        "engine": "FastEmbed + Qwen2.5-7B"
    }), 200


@app.route("/chat", methods=["POST"])
def chat():
    """RAG-enabled Chat completion endpoint."""
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        # Load DB lazily upon first incoming user request
        db = get_vector_db()

        # Retrieve top 5 most contextually relevant chunks
        relevant_docs = db.similarity_search(user_message, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # Construct System Prompt with Knowledge Base boundary limits
        system_prompt = f"""You are the official AI Assistant for the PMI Chennai Chapter.
Answer the user's question accurately using ONLY the retrieved facts from our knowledge base context below.
If a specific name, role, or detail is not present in the context, politely state that you do not have that specific information.

=== RETRIEVED KNOWLEDGE BASE CONTEXT ===
{context_text}
=========================================="""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Query Hugging Face Serverless Inference API
        response = client.chat_completion(
            messages,
            max_tokens=300,
            temperature=0.1
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply}), 200

    except Exception as e:
        logger.error(f"Error handling chat request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Dynamic port assignment required by Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
