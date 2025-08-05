from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# === Load label2id and create id2label ===
with open('label2id.json', 'r') as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# === Load model and tokenizer ===
model_path = "text_classifier_model.pkl"  # despite .pkl, it's a HF model folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# === FastAPI app ===
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    text = input.text.strip()
    if not text:
        return {"error": "Input text is empty."}
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probs, dim=-1).item()
        predicted_label = id2label[predicted_id]
    return {
        "predicted_label": predicted_label,
        "predicted_id": predicted_id,
        "probabilities": probs.tolist()[0]
    }