import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)  # Enables frontend requests without CORS errors

# Fetch token securely from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

# Load Knowledge Base
KNOWLEDGE_BASE = ""
if os.path.exists("qa_data.json"):
    with open("qa_data.json", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = f.read()

SYSTEM_PROMPT = f"""You are the official assistant for our non-profit organization.
Answer questions using ONLY facts in the Knowledge Base provided below. If not present, state that you do not have that information.

=== KNOWLEDGE BASE ===
{KNOWLEDGE_BASE}
======================
"""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat_completion(
            messages,
            max_tokens=300,
            temperature=0.1
        )
        
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
