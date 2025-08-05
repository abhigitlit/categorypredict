# Test Your Trained Text Classification Model
# Run this after you've trained your model using the main training script

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=== TEXT CLASSIFICATION MODEL TESTER ===\n")

# Method 1: Load saved model (if you saved it)
def load_saved_model():
    """Load the model and vectorizer that were saved during training"""
    try:
        model = joblib.load('text_classifier_model.pkl')
        vectorizer = joblib.load('text_vectorizer.pkl')
        print("✅ Successfully loaded saved model!")
        return model, vectorizer
    except FileNotFoundError:
        print("❌ Saved model files not found. Please run the training script first.")
        return None, None

# Method 2: Single text classification
def classify_single_text(model, vectorizer, text):
    """Classify a single piece of text"""
    if not text.strip():
        return "Empty text", 0.0
    
    # Transform the text
    text_vectorized = vectorizer.transform([text])
    
    # Get prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Get confidence scores for all categories
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = max(probabilities)
    
    # Get all category probabilities
    all_probs = {}
    for i, category in enumerate(model.classes_):
        all_probs[category] = probabilities[i]
    
    return prediction, confidence, all_probs

# Method 3: Batch testing with multiple texts
def test_multiple_texts(model, vectorizer, texts, expected_labels=None):
    """Test multiple texts at once"""
    results = []
    
    print("=== BATCH TESTING RESULTS ===")
    print("-" * 80)
    
    for i, text in enumerate(texts):
        prediction, confidence, all_probs = classify_single_text(model, vectorizer, text)
        
        result = {
            'text': text,
            'predicted': prediction,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
        
        if expected_labels:
            result['expected'] = expected_labels[i]
            result['correct'] = prediction == expected_labels[i]
        
        results.append(result)
        
        # Print result
        print(f"Text {i+1}: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        print(f"Predicted: {prediction} (Confidence: {confidence:.2%})")
        
        if expected_labels:
            correct_symbol = "✅" if result['correct'] else "❌"
            print(f"Expected: {expected_labels[i]} {correct_symbol}")
        
        # Show top 3 probabilities
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        print("Top predictions:")
        for category, prob in sorted_probs[:3]:
            print(f"  {category}: {prob:.2%}")
        print("-" * 80)
    
    return results

# Method 4: Interactive testing
def interactive_testing(model, vectorizer):
    """Interactive mode where you can type text and get predictions"""
    print("=== INTERACTIVE TESTING MODE ===")
    print("Type text to classify (or 'quit' to exit):\n")
    
    while True:
        user_input = input("Enter text to classify: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter some text!")
            continue
        
        prediction, confidence, all_probs = classify_single_text(model, vectorizer, user_input)
        
        print(f"\n🔍 Analysis of: '{user_input}'")
        print(f"📊 Predicted Category: {prediction}")
        print(f"🎯 Confidence: {confidence:.2%}")
        
        print("\n📈 All Category Probabilities:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for category, prob in sorted_probs:
            bar = "█" * int(prob * 20)  # Simple progress bar
            print(f"  {category:15}: {prob:.2%} {bar}")
        print()

# Method 5: Test with CSV file
def test_with_csv(model, vectorizer, csv_file, text_column, label_column=None):
    """Test the model using a CSV file"""
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded test file: {csv_file}")
        print(f"📊 Number of test samples: {len(df)}")
        
        texts = df[text_column].fillna("").tolist()
        expected_labels = df[label_column].tolist() if label_column and label_column in df.columns else None
        
        results = test_multiple_texts(model, vectorizer, texts, expected_labels)
        
        if expected_labels:
            # Calculate accuracy
            correct = sum(1 for r in results if r['correct'])
            accuracy = correct / len(results)
            print(f"\n📈 Overall Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
            
            # Show confusion matrix
            predicted = [r['predicted'] for r in results]
            plot_confusion_matrix(expected_labels, predicted, model.classes_)
        
        return results
        
    except FileNotFoundError:
        print(f"❌ File {csv_file} not found!")
        return None
    except KeyError as e:
        print(f"❌ Column {e} not found in CSV file!")
        return None

# Method 6: Visualize results
def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Main testing function
def main():
    """Main testing function"""
    
    # Load the model
    model, vectorizer = load_saved_model()
    if model is None:
        return
    
    print(f"Model loaded! It can classify into these categories:")
    for i, category in enumerate(model.classes_):
        print(f"  {i+1}. {category}")
    print()
    
    # Example test texts (replace with your own examples)
    example_texts = [
        "This is a sample text for testing",
        "Another example to test the classifier",
        "Yet another piece of text to classify"
    ]
    
    # Expected labels (optional - replace with actual expected results)
    expected_labels = None  # ["category1", "category2", "category3"]
    
    while True:
        print("Choose testing method:")
        print("1. Test example texts")
        print("2. Interactive testing (type your own text)")
        print("3. Test with CSV file")
        print("4. Single text test")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n" + "="*50)
            test_multiple_texts(model, vectorizer, example_texts, expected_labels)
            
        elif choice == "2":
            print("\n" + "="*50)
            interactive_testing(model, vectorizer)
            
        elif choice == "3":
            csv_file = input("Enter CSV filename: ").strip()
            text_col = input("Enter text column name: ").strip()
            label_col = input("Enter label column name (or press Enter if none): ").strip()
            label_col = label_col if label_col else None
            
            print("\n" + "="*50)
            test_with_csv(model, vectorizer, csv_file, text_col, label_col)
            
        elif choice == "4":
            text = input("Enter text to classify: ").strip()
            prediction, confidence, all_probs = classify_single_text(model, vectorizer, text)
            
            print(f"\n🔍 Result:")
            print(f"📊 Predicted: {prediction}")
            print(f"🎯 Confidence: {confidence:.2%}")
            print(f"📈 All probabilities: {all_probs}")
            
        elif choice == "5":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice! Please enter 1-5.")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

# Additional utility functions

def quick_test(text_list):
    """Quick function to test a list of texts"""
    model, vectorizer = load_saved_model()
    if model is None:
        return
    
    results = []
    for text in text_list:
        prediction, confidence, _ = classify_single_text(model, vectorizer, text)
        results.append({'text': text, 'prediction': prediction, 'confidence': confidence})
    
    return results

def get_model_info():
    """Get information about the loaded model"""
    model, vectorizer = load_saved_model()
    if model is None:
        return
    
    print("=== MODEL INFORMATION ===")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of categories: {len(model.classes_)}")
    print(f"Categories: {list(model.classes_)}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Max features: {vectorizer.max_features}")

# Example usage:
# Uncomment these lines to run specific tests

# Test specific examples
# results = quick_test([
#     "Your first test text here",
#     "Your second test text here"
# ])
# print(results)

# Get model information
# get_model_info()