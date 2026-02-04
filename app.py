import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

app = Flask(__name__)
CORS(app)

MODEL_ID = "fahmi553/anonymous-talk-sentiment"
HF_TOKEN = os.environ.get("HF_TOKEN")

print("Loading model… this may take a minute on cold start")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN
)

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device
)

print("Model loaded successfully")


@app.route('/')
def home():
    return jsonify({"status": "Local AI model running"}), 200


@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        result = classifier(text)[0]

        label = result["label"]
        score = float(result["score"])

        # Optional label mapping if your model uses LABEL_0 style
        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "POSITIVE"
        }

        label = label_map.get(label, label)

        return jsonify({
            "status": "success",
            "result": label,
            "confidence": score
        })

    except Exception as e:
        print("INFERENCE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
