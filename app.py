import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
MODEL_NAME = "fahmi553/anonymous-talk-model"

print(f"⏳ Loading AI Model: {MODEL_NAME}...")

try:
    classifier = pipeline("text-classification", model=MODEL_NAME)
    print("✅ AI Ready!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️ Falling back to default model...")
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = classifier(text, truncation=True, max_length=512)[0]

        label = result['label']
        score = result['score']

        print(f"🔍 Prediction: {label} ({score:.4f})")

        return jsonify({
            "status": "success",
            "result": label,
            "confidence": score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
