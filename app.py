import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Use the new required router URL
API_URL = "https://router.huggingface.co/hf-inference/models/fahmi553/anonymous-talk-sentiment"
HF_TOKEN = os.environ.get("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.route('/')
def home():
    return jsonify({"status": "AI Service is Running"}), 200

# NEW: Route to test if your HF_TOKEN is actually working
@app.route('/test-token')
def test_token():
    try:
        res = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
        return jsonify(res.json()), res.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        # Check if response is empty to avoid the "line 1 column 1" error
        if not response.text:
            return jsonify({"error": "Empty response from HF", "code": response.status_code}), 500

        ai_response = response.json()
        
        if response.status_code != 200:
            return jsonify({"status": "error", "details": ai_response}), response.status_code

        if isinstance(ai_response, list) and len(ai_response) > 0:
            inner = ai_response[0]
            top_result = inner[0] if isinstance(inner, list) else inner
            return jsonify({
                "status": "success",
                "result": top_result['label'],
                "confidence": top_result['score']
            })

        return jsonify({"error": "Unexpected format", "raw": ai_response}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
