import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_URL = "https://router.huggingface.co/hf-inference/models/fahmi553/anonymous-talk-sentiment"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


@app.route('/')
def home():
    return jsonify({"status": "AI Service is Running"}), 200


@app.route('/test-token')
def test_token():
    try:
        res = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers=headers,
            timeout=20
        )
        return jsonify(res.json()), res.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get('text', '').strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }

        hf_response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        print("HF STATUS:", hf_response.status_code)
        print("HF RESPONSE TEXT (first 500 chars):")
        print(hf_response.text[:500])

        if hf_response.status_code != 200:
            return jsonify({
                "status": "error",
                "message": "HF request failed",
                "hf_status": hf_response.status_code,
                "raw": hf_response.text
            }), 500

        try:
            ai_response = hf_response.json()
        except Exception:
            return jsonify({
                "status": "error",
                "message": "HF did not return JSON",
                "raw": hf_response.text
            }), 500

        top_result = None

        if isinstance(ai_response, list) and len(ai_response) > 0:
            first = ai_response[0]

            if isinstance(first, list) and len(first) > 0:
                top_result = first[0]
            elif isinstance(first, dict):
                top_result = first

        if not top_result or 'label' not in top_result or 'score' not in top_result:
            return jsonify({
                "status": "error",
                "message": "Unexpected HF output format",
                "parsed": ai_response
            }), 500

        return jsonify({
            "status": "success",
            "result": top_result['label'],
            "confidence": float(top_result['score'])
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "HF request timeout"}), 504

    except Exception as e:
        print("ANALYZE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
