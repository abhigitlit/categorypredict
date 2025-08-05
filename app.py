# Text Classification Model Hosting with FastAPI
# Complete web application to host your trained model

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class TextInput(BaseModel):
    text: str
    return_probabilities: bool = True

class PredictionResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float
    all_probabilities: Dict[str, float]
    timestamp: str

class BatchTextInput(BaseModel):
    texts: List[str]
    return_probabilities: bool = True

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int

class FeedbackInput(BaseModel):
    text: str
    predicted_category: str
    correct_category: str
    confidence: float
cat2label = {
  "Average": 0,
  "Profit and Loss": 1,
  "Percentage": 2,
  "Work and Time": 3,
  "Problems on Ages": 4,
  "Permutation & Combination": 5,
  "Number Series": 6,
  "Simplification": 7,
  "LCM & HCF": 8,
  "Partnership": 9,
  "Approximation": 10,
  "Simple Interest": 11,
  "Speed and Distance": 12,
  "Boats and Streams": 13,
  "Mixtures and Alligations": 14,
  "Ratio and Proportion": 15,
  "Algebra": 16,
  "Probability": 17
}
label2cat = {value: key for key, value in cat2label.items()}

# Text Classification Model Service
class ModelService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_info = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            self.model = joblib.load('text_classifier_model.pkl')
            self.vectorizer = joblib.load('text_vectorizer.pkl')
            
            # Store model information
            self.model_info = {
                "model_type": type(self.model).__name__,
                "categories": list(self.model.classes_),
                "num_categories": len(self.model.classes_),
                "vocabulary_size": len(self.vectorizer.get_feature_names_out()),
                "loaded_at": datetime.now().isoformat()
            }
            
            logger.info(f"Model loaded successfully: {self.model_info['model_type']}")
            logger.info(f"Categories: {self.model_info['categories']}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text: str, return_probabilities: bool = True):
        """Make prediction for a single text"""
        if not text.strip():
            raise ValueError("Empty text provided")
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([text])
            
            # Get prediction
            prediction = self.model.predict(text_vectorized)[0]
            
            # Get probabilities
            probabilities = self.model.predict_proba(text_vectorized)[0]
            confidence = float(max(probabilities))
            all_probs = {}
            if return_probabilities:
                for i, category in enumerate(self.model.classes_):
                    all_probs[label2cat[category]] = float(probabilities[i])
            top_5_probs = dict(list(all_probs.items())[:5])
            return {
                "predicted_category": label2cat[prediction],
                "confidence": confidence,
                "all_probabilities": top_5_probs
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, texts: List[str], return_probabilities: bool = True):
        """Make predictions for multiple texts"""
        if not texts:
            raise ValueError("No texts provided")
        
        results = []
        for text in texts:
            try:
                result = self.predict(text, return_probabilities)
                result["text"] = text
                result["timestamp"] = datetime.now().isoformat()
                results.append(result)
            except Exception as e:
                # Add error result for failed predictions
                results.append({
                    "text": text,
                    "predicted_category": "ERROR",
                    "confidence": 0.0,
                    "all_probabilities": {},
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def get_model_info(self):
        """Get model information"""
        return self.model_info
    
    def save_feedback(self, feedback_data):
        """Save user feedback for model improvement"""
        try:
            # Load existing feedback
            feedback_file = "user_feedback.json"
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    existing_feedback = json.load(f)
            else:
                existing_feedback = []
            
            # Add timestamp and append new feedback
            feedback_data["timestamp"] = datetime.now().isoformat()
            existing_feedback.append(feedback_data)
            
            # Save back to file
            with open(feedback_file, 'w') as f:
                json.dump(existing_feedback, f, indent=2)
            
            logger.info("Feedback saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False

# Initialize FastAPI app and model service
app = FastAPI(
    title="Text Classification API",
    description="API for hosting trained text classification model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service
model_service = ModelService()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Classification API</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-bottom: 20px;
            }
            h1 {
                color: #4a5568;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            h2 {
                color: #2d3748;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 10px;
            }
            .input-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #4a5568;
            }
            textarea, input {
                width: 100%;
                padding: 12px;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            textarea:focus, input:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s;
                margin-right: 10px;
                margin-bottom: 10px;
            }
            button:hover {
                transform: translateY(-2px);
            }
            .result {
                background: #f7fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
            }
            .prediction {
                font-size: 1.2em;
                font-weight: bold;
                color: #2d3748;
                margin-bottom: 10px;
            }
            .confidence {
                margin-bottom: 15px;
            }
            .confidence-bar {
                background: #e2e8f0;
                height: 20px;
                border-radius: 10px;
                overflow: hidden;
                margin-top: 5px;
            }
            .confidence-fill {
                background: linear-gradient(90deg, #667eea, #764ba2);
                height: 100%;
                transition: width 0.5s ease;
            }
            .probabilities {
                margin-top: 15px;
            }
            .prob-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                padding: 8px;
                background: white;
                border-radius: 5px;
            }
            .prob-bar {
                width: 100px;
                height: 10px;
                background: #e2e8f0;
                border-radius: 5px;
                overflow: hidden;
                margin-left: 10px;
            }
            .prob-fill {
                height: 100%;
                background: #667eea;
                transition: width 0.3s ease;
            }
            .feedback-section {
                background: #fff5f5;
                border: 1px solid #fed7d7;
                border-radius: 8px;
                padding: 15px;
                margin-top: 15px;
            }
            .feedback-buttons {
                margin-top: 10px;
            }
            .correct-btn {
                background: #48bb78;
            }
            .incorrect-btn {
                background: #f56565;
            }
            .model-info {
                background: #ebf8ff;
                border: 1px solid #bee3f8;
                border-radius: 8px;
                padding: 15px;
            }
            .loading {
                opacity: 0.6;
                pointer-events: none;
            }
            .error {
                background: #fed7d7;
                color: #c53030;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Text Classification API</h1>
            
            <div class="model-info" id="modelInfo">
                <h2>📊 Model Information</h2>
                <p>Loading model information...</p>
            </div>
        </div>

        <div class="container">
            <h2>🔍 Single Text Classification</h2>
            <div class="input-group">
                <label for="textInput">Enter text to classify:</label>
                <textarea id="textInput" rows="4" placeholder="Type your text here..."></textarea>
            </div>
            <button onclick="classifyText()">Classify Text</button>
            <button onclick="clearSingle()">Clear</button>
            
            <div id="singleResult" class="result" style="display: none;">
                <!-- Results will appear here -->
            </div>
        </div>

        <div class="container">
            <h2>📋 Batch Classification</h2>
            <div class="input-group">
                <label for="batchInput">Enter multiple texts (one per line):</label>
                <textarea id="batchInput" rows="6" placeholder="Text 1&#10;Text 2&#10;Text 3&#10;..."></textarea>
            </div>
            <button onclick="classifyBatch()">Classify Batch</button>
            <button onclick="clearBatch()">Clear</button>
            
            <div id="batchResult" class="result" style="display: none;">
                <!-- Batch results will appear here -->
            </div>
        </div>

        <script>
            // Load model information on page load
            window.onload = function() {
                loadModelInfo();
            };

            async function loadModelInfo() {
                try {
                    const response = await fetch('/model/info');
                    const info = await response.json();
                    
                    document.getElementById('modelInfo').innerHTML = `
                        <h2>📊 Model Information</h2>
                        <p><strong>Model Type:</strong> ${info.model_type}</p>
                        <p><strong>Categories:</strong> ${info.categories.join(', ')}</p>
                        <p><strong>Number of Categories:</strong> ${info.num_categories}</p>
                        <p><strong>Vocabulary Size:</strong> ${info.vocabulary_size.toLocaleString()}</p>
                        <p><strong>Loaded At:</strong> ${new Date(info.loaded_at).toLocaleString()}</p>
                    `;
                } catch (error) {
                    document.getElementById('modelInfo').innerHTML = `
                        <h2>📊 Model Information</h2>
                        <p class="error">Error loading model information: ${error.message}</p>
                    `;
                }
            }

            async function classifyText() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) {
                    alert('Please enter some text to classify!');
                    return;
                }

                const resultDiv = document.getElementById('singleResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>🔄 Classifying...</p>';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            return_probabilities: true
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    displaySingleResult(result, text);

                } catch (error) {
                    resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                }
            }

            function displaySingleResult(result, originalText) {
                const resultDiv = document.getElementById('singleResult');
                const confidence = (result.confidence * 100).toFixed(1);
                
                let probsHtml = '';
                if (result.all_probabilities) {
                    // Sort probabilities by value
                    const sortedProbs = Object.entries(result.all_probabilities)
                        .sort(([,a], [,b]) => b - a);
                    
                    probsHtml = '<div class="probabilities"><h3>All Probabilities:</h3>';
                    sortedProbs.forEach(([category, prob]) => {
                        const percentage = (prob * 100).toFixed(1);
                        probsHtml += `
                            <div class="prob-item">
                                <span>${category}: ${percentage}%</span>
                                <div class="prob-bar">
                                    <div class="prob-fill" style="width: ${percentage}%"></div>
                                </div>
                            </div>
                        `;
                    });
                    probsHtml += '</div>';
                }

                resultDiv.innerHTML = `
                    <div class="prediction">
                        🎯 Predicted Category: <span style="color: #667eea;">${result.predicted_category}</span>
                    </div>
                    <div class="confidence">
                        Confidence: ${confidence}%
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                    ${probsHtml}
                    <div class="feedback-section">
                        <p><strong>Was this prediction correct?</strong></p>
                        <div class="feedback-buttons">
                            <button class="correct-btn" onclick="sendFeedback('${originalText}', '${result.predicted_category}', '${result.predicted_category}', ${result.confidence})">
                                ✅ Correct
                            </button>
                            <button class="incorrect-btn" onclick="promptCorrection('${originalText}', '${result.predicted_category}', ${result.confidence})">
                                ❌ Incorrect
                            </button>
                        </div>
                    </div>
                `;
            }

            async function classifyBatch() {
                const batchText = document.getElementById('batchInput').value.trim();
                if (!batchText) {
                    alert('Please enter texts to classify!');
                    return;
                }

                const texts = batchText.split('\\n').filter(text => text.trim());
                if (texts.length === 0) {
                    alert('No valid texts found!');
                    return;
                }

                const resultDiv = document.getElementById('batchResult');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `<p>🔄 Classifying ${texts.length} texts...</p>`;

                try {
                    const response = await fetch('/predict/batch', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            texts: texts,
                            return_probabilities: true
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    displayBatchResults(result);

                } catch (error) {
                    resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                }
            }

            function displayBatchResults(result) {
                const resultDiv = document.getElementById('batchResult');
                let html = `<h3>📊 Batch Results (${result.total_processed} texts processed)</h3>`;

                result.predictions.forEach((pred, index) => {
                    const confidence = (pred.confidence * 100).toFixed(1);
                    const textPreview = pred.text.length > 100 ? pred.text.substring(0, 100) + '...' : pred.text;
                    
                    html += `
                        <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; margin: 10px 0; background: white;">
                            <p><strong>Text ${index + 1}:</strong> "${textPreview}"</p>
                            <p><strong>Prediction:</strong> <span style="color: #667eea;">${pred.predicted_category}</span></p>
                            <p><strong>Confidence:</strong> ${confidence}%</p>
                            <div class="confidence-bar" style="width: 200px;">
                                <div class="confidence-fill" style="width: ${confidence}%"></div>
                            </div>
                        </div>
                    `;
                });

                resultDiv.innerHTML = html;
            }

            function promptCorrection(text, predicted, confidence) {
                const correctCategory = prompt(`What should the correct category be for this text?\\n\\nText: "${text}"\\n\\nPredicted: ${predicted}`);
                
                if (correctCategory && correctCategory.trim()) {
                    sendFeedback(text, predicted, correctCategory.trim(), confidence);
                }
            }

            async function sendFeedback(text, predicted, correct, confidence) {
                try {
                    const response = await fetch('/feedback', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            predicted_category: predicted,
                            correct_category: correct,
                            confidence: confidence
                        })
                    });

                    if (response.ok) {
                        alert('✅ Thank you for your feedback!');
                    } else {
                        alert('❌ Error saving feedback');
                    }
                } catch (error) {
                    alert('❌ Error sending feedback: ' + error.message);
                }
            }

            function clearSingle() {
                document.getElementById('textInput').value = '';
                document.getElementById('singleResult').style.display = 'none';
            }

            function clearBatch() {
                document.getElementById('batchInput').value = '';
                document.getElementById('batchResult').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_service.get_model_info()

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Predict category for a single text"""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = model_service.predict(input_data.text, input_data.return_probabilities)
        
        return PredictionResponse(
            text=input_data.text,
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """Predict categories for multiple texts"""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(input_data.texts) > 1000:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 1000)")
    
    try:
        results = model_service.predict_batch(input_data.texts, input_data.return_probabilities)
        
        predictions = []
        for result in results:
            predictions.append(PredictionResponse(
                text=result["text"],
                predicted_category=result["predicted_category"],
                confidence=result["confidence"],
                all_probabilities=result.get("all_probabilities", {}),
                timestamp=result["timestamp"]
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackInput):
    """Submit feedback for model improvement"""
    try:
        feedback_data = {
            "text": feedback.text,
            "predicted_category": feedback.predicted_category,
            "correct_category": feedback.correct_category,
            "confidence": feedback.confidence
        }
        
        success = model_service.save_feedback(feedback_data)
        
        if success:
            return {"message": "Feedback saved successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save feedback")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics"""
    try:
        if os.path.exists("user_feedback.json"):
            with open("user_feedback.json", 'r') as f:
                feedback_data = json.load(f)
            
            total_feedback = len(feedback_data)
            correct_predictions = sum(1 for f in feedback_data 
                                    if f["predicted_category"] == f["correct_category"])
            
            return {
                "total_feedback": total_feedback,
                "correct_predictions": correct_predictions,
                "accuracy": correct_predictions / total_feedback if total_feedback > 0 else 0,
                "last_feedback": feedback_data[-1]["timestamp"] if feedback_data else None
            }
        else:
            return {"total_feedback": 0, "correct_predictions": 0, "accuracy": 0}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback stats: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Text Classification API Server...")
    print("📊 Make sure your model files are in the same directory:")
    print("   - text_classifier_model.pkl")
    print("   - text_vectorizer.pkl")
    print("\n🌐 The web interface will be available at: http://localhost:8000")
    print("📖 API documentation will be at: http://localhost:8000/docs")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        reload=False,
        log_level="info"
    )