import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Load label2id and create id2label ===
with open('label2id.json', 'r') as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# === Load model and tokenizer ===
model_path = "text_classifier_model.pkl"  # Replace with your model directory or checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
while True:
    text = input("Enter text to classify (or 'quit' to exit): ").strip()
    if text.lower() == 'quit':
        break
    if not text:
        print("Empty input. Try again.")
        continue
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probs, dim=-1).item()
        predicted_label = id2label[predicted_id]

    print(f"Predicted label: {predicted_label} (id={predicted_id})\n")
