import json
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Dict, Any
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries and provide fallbacks
try:
    import spacy
    from spacy.training import Example
    from spacy.tokens import DocBin
    from spacy.lang.en import English
except ImportError:
    logger.error("spaCy is not installed. Run 'pip install spacy' and download a model (e.g., 'en_core_web_sm').")
    sys.exit(1)

try:
    from transformers import pipeline, DistilBertForQuestionAnswering, DistilBertTokenizerFast, Trainer, TrainingArguments
    from datasets import Dataset
    import torch
except ImportError:
    logger.error("Transformers or datasets library is missing. Install with 'pip install transformers datasets torch'.")
    sys.exit(1)

try:
    from fuzzywuzzy import process
except ImportError:
    logger.error("fuzzywuzzy is not installed. Install with 'pip install fuzzywuzzy'.")
    sys.exit(1)

try:
    from flask import Flask, request, jsonify, render_template, send_from_directory
    from flask_cors import CORS
except ImportError:
    logger.error("Flask or Flask-CORS is not installed. Install with 'pip install flask flask-cors'.")
    sys.exit(1)

# Configuration Management
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "disease_file": "diseases.json",
    "output_dir": "output",
    "spacy_model": "en_core_web_sm",
    "qa_model": "distilbert-base-uncased-distilled-squad",
    "training_data_file": "training_data.spacy",
    "config_path": "base_config.cfg"
}

def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration from a JSON file or return default config."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG

config = load_config()

# Initialize Flask app with CSP
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Add CSP header to all responses with relaxed policies for development
@app.after_request
def add_security_headers(response):
    # Relaxed CSP for development; adjust for production (e.g., remove 'unsafe-inline')
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "  # Allow inline scripts temporarily for debugging
        "style-src 'self' 'unsafe-inline'; "   # Allow inline styles temporarily
        "connect-src 'self'; "
        "media-src 'self' data:;"              # Allow data URIs for media
    )
    return response

# Load spaCy and QA models
try:
    nlp = spacy.load(config["spacy_model"])
except OSError:
    logger.warning(f"SpaCy model '{config['spacy_model']}' not found. Downloading...")
    spacy.cli.download(config["spacy_model"])
    nlp = spacy.load(config["spacy_model"])

qa_pipeline = pipeline("question-answering", model=config["qa_model"], tokenizer=config["qa_model"])

# Load disease data
def load_diseases(file_path: str = config["disease_file"]) -> Dict[str, Any]:
    """Load disease data from a JSON file."""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Disease file '{file_path}' not found.")
        return {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict) or "diseases" not in data:
                raise ValueError("Invalid JSON structure. Expected a 'diseases' key.")
            return data
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error loading '{file_path}': {e}")
        return {}

diseases_data = load_diseases()
diseases = diseases_data.get("diseases", [])

# Symptom Extraction
def extract_symptoms(user_input: str) -> List[str]:
    """Extract symptoms from user input using spaCy NER."""
    doc = nlp(user_input)
    symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
    if not symptoms:
        symptoms = [sym.strip() for sym in user_input.split(",") if sym.strip()]
    return symptoms

# Disease Matching
def match_disease_fuzzy(symptoms: List[str], diseases: List[Dict], threshold: int = 1, score_cutoff: int = 60) -> Dict[str, Any]:
    """Match diseases based on symptoms using fuzzy matching and return table data."""
    if not diseases:
        return {"table": {"headers": ["Message"], "rows": [["No disease data available."]]}}
    if not symptoms:
        return {"table": {"headers": ["Message"], "rows": [["No symptoms provided."]]}}

    matched_diseases = []
    for disease in diseases:
        disease_symptoms = [s.lower() for s in disease["symptoms"]]
        match_count = sum(1 for s in symptoms if process.extractOne(s.lower(), disease_symptoms, score_cutoff=score_cutoff))
        if match_count >= threshold:
            matched_diseases.append((disease["name"], disease["symptoms"], match_count))

    if not matched_diseases:
        return {"table": {"headers": ["Message"], "rows": [["No matching diseases found."]]}}

    matched_diseases.sort(key=lambda x: x[2], reverse=True)
    headers = ["Disease Name", "Matches", "Symptoms"]
    rows = [[name, str(count), ", ".join(syms)] for name, syms, count in matched_diseases[:5]]
    return {"table": {"headers": headers, "rows": rows}}

# Question Answering
def answer_question(user_input: str, diseases: List[Dict]) -> Dict[str, Any]:
    """Answer user questions using the QA pipeline and return table data."""
    context = " ".join(
        f"{d['name']} is a {d['type']}. Symptoms: {', '.join(d['symptoms'])}. "
        f"Causes: {d['causes']}. Treatment: {d['treatment']}. Prevention: {d.get('prevention', 'N/A')}."
        for d in diseases
    )
    result = qa_pipeline(question=user_input, context=context)
    answer = result["answer"] if result else "I couldn't find an answer."
    return {"table": {"headers": ["Question", "Answer"], "rows": [[user_input, answer]]}}

# Training NER Model
def train_ner_model(training_data: List[tuple], output_dir: str = config["output_dir"]):
    """Train a spaCy NER model."""
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label("SYMPTOM")

    doc_bin = DocBin()
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        doc_bin.add(example.reference)

    output_path = Path(output_dir) / config["training_data_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc_bin.to_disk(output_path)

    config_path = Path(config["config_path"])
    if not config_path.exists():
        logger.error(f"Config file '{config_path}' not found. Please create it using 'python -m spacy init config'.")
        return

    try:
        subprocess.run(
            ["python", "-m", "spacy", "train", str(config_path), "--output", output_dir],
            check=True
        )
        logger.info(f"NER model trained and saved in '{output_dir}/model-best'")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during NER training: {e}")

# Fine-Tune QA Model
def fine_tune_qa_model():
    """Fine-tune the DistilBERT QA model."""
    if not diseases:
        logger.error("No disease data available for training.")
        return

    contexts, questions, answers = [], [], []
    for disease in diseases:
        context = f"{disease['name']} is a {disease['type']}. Symptoms: {', '.join(disease['symptoms'])}."
        symptoms_str = ", ".join(disease["symptoms"])
        start_idx = context.find(symptoms_str)
        if start_idx == -1:
            continue
        end_idx = start_idx + len(symptoms_str)
        contexts.append(context)
        questions.append(f"What are the symptoms of {disease['name']}?")
        answers.append({"text": symptoms_str, "start": start_idx, "end": end_idx})

    tokenizer = DistilBertTokenizerFast.from_pretrained(config["qa_model"])
    dataset = Dataset.from_dict({"context": contexts, "question": questions, "answers": answers})

    def preprocess(example):
        tokenized = tokenizer(example["question"], example["context"], truncation=True, padding="max_length", max_length=512)
        start = example["answers"]["start"]
        end = example["answers"]["end"]
        tokenized["start_positions"] = tokenized.char_to_token(0, start) or 0
        tokenized["end_positions"] = tokenized.char_to_token(0, end) or 0
        return tokenized

    dataset = dataset.map(preprocess)
    model = DistilBertForQuestionAnswering.from_pretrained(config["qa_model"])

    training_args = TrainingArguments(
        output_dir="./qa_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained("./fine_tuned_qa_model")
    tokenizer.save_pretrained("./fine_tuned_qa_model")
    logger.info("QA model fine-tuned and saved.")

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip().lower()
    if not user_message:
        return jsonify({'table': {'headers': ['Message'], 'rows': [['Please provide a message.']]}}), 400

    if not diseases:
        return jsonify({'table': {'headers': ['Message'], 'rows': [['Disease data unavailable.']]}}), 500

    for disease in diseases:
        if disease['name'].lower() in user_message:
            reply = {
                'table': {
                    'headers': ['Field', 'Details'],
                    'rows': [
                        ['Name', disease['name']],
                        ['Type', disease['type']],
                        ['Symptoms', ', '.join(disease['symptoms'])],
                        ['Causes', disease['causes']],
                        ['Treatment', disease['treatment']],
                        ['Prevention', disease.get('prevention', 'N/A')]
                    ]
                }
            }
            return jsonify(reply)

    if "symptom" in user_message:
        symptoms = extract_symptoms(user_message)
        reply = match_disease_fuzzy(symptoms, diseases) if symptoms else {"table": {"headers": ["Message"], "rows": [["No symptoms detected."]]}}
    else:
        reply = answer_question(user_message, diseases)

    return jsonify(reply)

@app.route('/static/<path:filename>')
def static_files(filename):
    try:
        logger.info(f"Serving static file: {filename}")
        return send_from_directory('static', filename)
    except Exception as e:
        logger.error(f"Error serving static file '{filename}': {e}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

if __name__ == "__main__":
    # Example training data (move this to a separate file if needed)
    training_data = [
        ("The patient has a fever and sore throat.", {"entities": [(17, 22, "SYMPTOM"), (27, 38, "SYMPTOM")]})
    ]
    
    # Uncomment to train models on startup
    # train_ner_model(training_data)
    # fine_tune_qa_model()
    
    app.run(debug=True, host="0.0.0.0", port=5000)
