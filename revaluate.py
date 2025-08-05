# Auto-CSV Interactive Model Retraining System
# Automatically loads examples from CSV and prompts for corrections!

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os
from datetime import datetime
import random

class AutoCSVModelTrainer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.training_data = []
        self.corrections_log = []
        self.csv_data = None
        self.current_index = 0
        
    def load_model(self):
        """Load the existing trained model"""
        try:
            self.model = joblib.load('text_classifier_model.pkl')
            self.vectorizer = joblib.load('text_vectorizer.pkl')
            print("✅ Model loaded successfully!")
            self.load_training_history()
            return True
        except FileNotFoundError:
            print("❌ Model files not found. Please train a model first.")
            return False
    
    def load_csv_data(self, csv_file=None, text_column=None, label_column=None):
        """Auto-detect and load CSV data"""
        
        # Auto-detect CSV file if not provided
        if not csv_file:
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if not csv_files:
                print("❌ No CSV files found in current directory!")
                return False
            elif len(csv_files) == 1:
                csv_file = csv_files[0]
                print(f"📁 Found CSV file: {csv_file}")
            else:
                print("📁 Multiple CSV files found:")
                for i, file in enumerate(csv_files, 1):
                    print(f"   {i}. {file}")
                choice = input("Enter number to select file: ").strip()
                try:
                    csv_file = csv_files[int(choice) - 1]
                except:
                    print("❌ Invalid choice!")
                    return False
        
        # Load the CSV
        try:
            self.csv_data = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(self.csv_data)} rows from {csv_file}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
        
        # Auto-detect columns
        columns = self.csv_data.columns.tolist()
        print(f"📋 Available columns: {columns}")
        
        # Auto-detect text column
        if not text_column:
            text_candidates = [col for col in columns if any(keyword in col.lower() 
                             for keyword in ['text', 'content', 'message', 'review', 'comment', 'description'])]
            
            if text_candidates:
                text_column = text_candidates[0]
                print(f"🔍 Auto-detected text column: '{text_column}'")
            else:
                print("🤔 Please specify the text column:")
                for i, col in enumerate(columns, 1):
                    print(f"   {i}. {col}")
                choice = input("Enter number: ").strip()
                try:
                    text_column = columns[int(choice) - 1]
                except:
                    print("❌ Invalid choice!")
                    return False
        
        # Auto-detect label column (optional)
        if not label_column:
            label_candidates = [col for col in columns if any(keyword in col.lower() 
                              for keyword in ['label', 'category', 'class', 'tag', 'type'])]
            
            if label_candidates:
                label_column = label_candidates[0]
                print(f"🏷️  Auto-detected label column: '{label_column}'")
                use_labels = input("Use existing labels for comparison? (y/n): ").strip().lower()
                if use_labels not in ['y', 'yes']:
                    label_column = None
            else:
                print("ℹ️  No label column detected - will work in prediction-only mode")
                label_column = None
        
        self.text_column = text_column
        self.label_column = label_column
        
        # Clean data
        self.csv_data = self.csv_data.dropna(subset=[text_column])
        print(f"✅ Ready with {len(self.csv_data)} examples!")
        
        # Load progress if exists
        self.load_progress()
        
        return True
    
    def load_training_history(self):
        """Load previous training data and corrections"""
        try:
            with open('training_history.json', 'r') as f:
                history = json.load(f)
                self.training_data = history.get('training_data', [])
                self.corrections_log = history.get('corrections_log', [])
            print(f"📚 Loaded {len(self.corrections_log)} previous corrections")
        except FileNotFoundError:
            print("📝 Starting fresh training session")
    
    def load_progress(self):
        """Load progress through CSV file"""
        try:
            with open('csv_progress.json', 'r') as f:
                progress = json.load(f)
                self.current_index = progress.get('current_index', 0)
            print(f"📍 Resuming from example {self.current_index + 1}")
        except FileNotFoundError:
            self.current_index = 0
            print("🆕 Starting from the beginning")
    
    def save_progress(self):
        """Save all progress"""
        # Save training history
        history = {
            'training_data': self.training_data,
            'corrections_log': self.corrections_log,
            'last_updated': datetime.now().isoformat()
        }
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save CSV progress
        progress = {
            'current_index': self.current_index,
            'total_examples': len(self.csv_data),
            'last_updated': datetime.now().isoformat()
        }
        with open('csv_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Save updated model
        joblib.dump(self.model, 'text_classifier_model.pkl')
        joblib.dump(self.vectorizer, 'text_vectorizer.pkl')
    
    def predict_with_confidence(self, text):
        """Get prediction and confidence for a text"""
        if not text.strip():
            return None, 0.0, {}
        
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        confidence = max(probabilities)
        
        all_probs = {}
        for i, category in enumerate(self.model.classes_):
            all_probs[category] = probabilities[i]
        
        return prediction, confidence, all_probs
    
    def retrain_model(self):
        """Retrain the model with accumulated corrections"""
        if not self.training_data:
            return False
        
        print(f"🔄 Retraining with {len(self.training_data)} examples...")
        
        texts = [item['text'] for item in self.training_data]
        labels = [item['label'] for item in self.training_data]
        
        # Full retraining
        text_vectorized = self.vectorizer.fit_transform(texts)
        
        if isinstance(self.model, MultinomialNB):
            new_model = MultinomialNB()
        else:
            new_model = LogisticRegression(max_iter=1000, random_state=42)
        
        new_model.fit(text_vectorized, labels)
        self.model = new_model
        
        print("✅ Model retrained!")
        return True
    
    def auto_training_session(self, mode='sequential'):
        """Main auto-training session with CSV data"""
        print("=== AUTO-CSV MODEL IMPROVEMENT SYSTEM ===")
        print("I'll show you examples from your CSV and you just tell me if I'm right!")
        print("Commands: 'skip', 'back', 'random', 'stats', 'retrain', 'quit'\n")
        
        session_corrections = 0
        session_tests = 0
        
        total_examples = len(self.csv_data)
        
        while self.current_index < total_examples:
            # Get current example
            row = self.csv_data.iloc[self.current_index]
            text = str(row[self.text_column]).strip()
            
            if not text:
                self.current_index += 1
                continue
            
            session_tests += 1
            
            # Show progress
            progress = (self.current_index + 1) / total_examples * 100
            print(f"\n{'='*60}")
            print(f"📍 Example {self.current_index + 1}/{total_examples} ({progress:.1f}%)")
            print(f"📝 Text: {text}")
            
            # Show actual label if available
            if self.label_column and self.label_column in row:
                actual_label = str(row[self.label_column])
                print(f"📋 Actual label: {actual_label}")
            
            # Get prediction
            prediction, confidence, all_probs = self.predict_with_confidence(text)
            
            if prediction is None:
                print("❌ Could not process this text")
                self.current_index += 1
                continue
            
            # Show prediction with visual confidence
            print(f"\n🤖 My Prediction: {prediction}")
            confidence_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
            print(f"🎯 Confidence: [{confidence_bar}] {confidence:.1%}")
            
            # Show top 3 predictions
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            print(f"\n📊 Top 3 predictions:")
            for i, (category, prob) in enumerate(sorted_probs[:3]):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                bar = "▓" * int(prob * 10)
                print(f"   {emoji} {category}: {prob:.1%} {bar}")
            
            # Simple prompt
            print(f"\n❓ Am I correct?")
            if self.label_column and self.label_column in row:
                actual = str(row[self.label_column]).strip()
                # Handle comparison safely for both text and numeric labels
                is_correct = str(prediction).lower() == actual.lower()
                print(f"   ✅ YES - I'm right!" if is_correct else f"   ❌ NO - Should be: {actual}")
                print(f"   ⏭️  SKIP this one")
            else:
                print(f"   ✅ YES - Correct!")
                print(f"   ❌ NO - Let me fix it")
                print(f"   ⏭️  SKIP this one")
            
            # Get user input
            response = input("Your answer (yes/no/skip/command): ").strip().lower()
            
            # Handle commands
            if response in ['quit', 'q', 'exit']:
                self.save_progress()
                print(f"\n📈 Session Summary:")
                print(f"   Examples reviewed: {session_tests}")
                print(f"   Corrections made: {session_corrections}")
                print(f"   Progress: {self.current_index + 1}/{total_examples}")
                print("   All progress saved! 💾")
                break
            
            elif response in ['skip', 's']:
                print("⏭️  Skipped!")
                self.current_index += 1
                continue
            
            elif response == 'back':
                if self.current_index > 0:
                    self.current_index -= 1
                    print("⬅️  Going back...")
                else:
                    print("❌ Already at the first example!")
                continue
            
            elif response == 'random':
                self.current_index = random.randint(0, total_examples - 1)
                print(f"🎲 Jumped to random example {self.current_index + 1}")
                continue
            
            elif response == 'stats':
                self.show_session_stats(session_tests, session_corrections)
                continue
            
            elif response == 'retrain':
                if self.training_data:
                    self.retrain_model()
                    print("🎯 Model updated!")
                else:
                    print("❌ No corrections to learn from yet!")
                continue
            
            # Handle yes/no responses
            elif response in ['yes', 'y', 'correct', '✅']:
                print("🎉 Great! Adding as positive example...")
                self.training_data.append({
                    'text': text,
                    'label': prediction,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'positive_reinforcement'
                })
            
            elif response in ['no', 'n', 'wrong', '❌']:
                # Get correct label
                if self.label_column and self.label_column in row:
                    correct_label = str(row[self.label_column]).strip()
                    print(f"📝 Using actual label: {correct_label}")
                else:
                    print(f"📚 Available categories: {list(self.model.classes_)}")
                    correct_label = input("Enter correct category: ").strip()
                
                # Convert to string for comparison
                if correct_label in [str(cls) for cls in self.model.classes_]:
                    # Log correction
                    correction = {
                        'text': text,
                        'predicted': str(prediction),
                        'correct': correct_label,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.corrections_log.append(correction)
                    
                    # Add to training data
                    self.training_data.append({
                        'text': text,
                        'label': correct_label,
                        'timestamp': datetime.now().isoformat(),
                        'type': 'correction'
                    })
                    
                    session_corrections += 1
                    print(f"📝 Correction saved: {prediction} → {correct_label}")
                else:
                    print(f"❌ '{correct_label}' is not a valid category!")
                    print(f"Valid categories: {list(self.model.classes_)}")
                    continue
            
            else:
                print("❌ Please answer: yes/no/skip or use commands")
                continue
            
            # Move to next example
            self.current_index += 1
            
            # Auto-retrain every 10 corrections
            if session_corrections > 0 and session_corrections % 10 == 0:
                print(f"\n🔄 Auto-retraining after {session_corrections} corrections...")
                self.retrain_model()
                print("✅ Model updated!")
            
            # Auto-save every 20 examples
            if session_tests % 20 == 0:
                self.save_progress()
                print("💾 Progress auto-saved!")
        
        # Final save and summary
        if self.current_index >= total_examples:
            print(f"\n🎊 COMPLETED ALL EXAMPLES!")
        
        self.save_progress()
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Total examples: {total_examples}")
        print(f"   Examples reviewed: {session_tests}")
        print(f"   Corrections made: {session_corrections}")
        print(f"   Training examples: {len(self.training_data)}")
        print("   Model saved! 🎯")
    
    def show_session_stats(self, tests, corrections):
        """Show current session statistics"""
        print(f"\n📊 SESSION STATS:")
        print(f"   Examples reviewed: {tests}")
        print(f"   Corrections made: {corrections}")
        print(f"   Accuracy so far: {((tests - corrections) / tests * 100):.1f%}" if tests > 0 else "   No tests yet")
        print(f"   Progress: {self.current_index + 1}/{len(self.csv_data)}")
        print(f"   Total training examples: {len(self.training_data)}")

def main():
    """Main function"""
    trainer = AutoCSVModelTrainer()
    
    # Load model
    if not trainer.load_model():
        print("Please train a model first!")
        return
    
    print(f"🎯 Model loaded with categories: {list(trainer.model.classes_)}")
    
    # Auto-load CSV
    csv_file = input("Enter CSV filename (or press Enter to auto-detect): ").strip()
    csv_file = csv_file if csv_file else None
    
    if not trainer.load_csv_data(csv_file):
        print("Failed to load CSV data!")
        return
    
    # Start auto-training session
    print(f"\n🚀 Starting auto-training session!")
    print(f"📋 I'll go through your CSV examples one by one")
    print(f"💡 Just tell me if I'm right or wrong - I'll learn from it!")
    
    trainer.auto_training_session()

if __name__ == "__main__":
    main()