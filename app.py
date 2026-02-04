import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_URL = "https://api-inference.huggingface.co/models/fahmi553/anonymous-talk-sentiment"
HF_TOKEN = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.route('/')
def home():
    return jsonify({"status": "AI Service is Running"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    payload = {"inputs": text}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        ai_response = response.json()
        if isinstance(ai_response, dict) and "error" in ai_response:
            return jsonify({"status": "loading", "message": ai_response["error"]}), 503
            
        if isinstance(ai_response, list) and len(ai_response) > 0:
            top_result = ai_response[0][0] if isinstance(ai_response[0], list) else ai_response[0]

            return jsonify({
                "status": "success",
                "result": top_result['label'],
                "confidence": top_result['score']
            })

        return jsonify({"error": "Unexpected API response format"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
