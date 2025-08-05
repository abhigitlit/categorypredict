# Text Classification for Beginners
# This code will help you train a model to classify text into categories

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load your data
print("=== STEP 1: Loading Your Data ===")
# Replace 'your_file.csv' with your actual file name
df = pd.read_csv('train_dataset.csv')

print(f"Dataset loaded! Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# Step 2: Examine your data
print("\n=== STEP 2: Understanding Your Data ===")

# You'll need to replace these column names with your actual column names
# Common names are: 'text', 'label', 'category', 'class', etc.
text_column = 'text'  # Replace with your text column name
label_column = 'label'  # Replace with your label column name

print(f"Number of unique categories: {df[label_column].nunique()}")
print(f"Categories: {df[label_column].unique()}")

print("\nCategory distribution:")
print(df[label_column].value_counts())

# Visualize category distribution
plt.figure(figsize=(10, 6))
df[label_column].value_counts().plot(kind='bar')
plt.title('Distribution of Categories')
plt.xlabel('Category')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 3: Prepare the data
print("\n=== STEP 3: Preparing Data for Machine Learning ===")

# Clean the data - remove any missing values
df = df.dropna(subset=[text_column, label_column])
print(f"After removing missing values: {df.shape}")

# Get our text and labels
X = df[text_column]  # The text we want to classify
y = df[label_column]  # The categories/labels

print(f"Total samples: {len(X)}")

# Step 4: Split data into training and testing sets
print("\n=== STEP 4: Splitting Data ===")
# We'll use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 5: Convert text to numbers (vectorization)
print("\n=== STEP 5: Converting Text to Numbers ===")
# Machine learning models need numbers, not text
# TF-IDF converts text to meaningful numbers

vectorizer = TfidfVectorizer(
    max_features=5000,  # Use top 5000 most important words
    stop_words='english',  # Remove common words like 'the', 'and'
    ngram_range=(1, 2),  # Use single words and pairs of words
    min_df=2  # Ignore words that appear less than 2 times
)

# Transform training data
X_train_vectorized = vectorizer.fit_transform(X_train)
print(f"Training data shape after vectorization: {X_train_vectorized.shape}")

# Transform test data (using the same vectorizer)
X_test_vectorized = vectorizer.transform(X_test)
print(f"Test data shape after vectorization: {X_test_vectorized.shape}")

# Step 6: Train different models
print("\n=== STEP 6: Training Models ===")

# Model 1: Naive Bayes (good for text classification)
print("Training Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Model 2: Logistic Regression (also very good for text)
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vectorized, y_train)

# Step 7: Test the models
print("\n=== STEP 7: Testing Models ===")

# Test Naive Bayes
nb_predictions = nb_model.predict(X_test_vectorized)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy:.3f} ({nb_accuracy*100:.1f}%)")

# Test Logistic Regression
lr_predictions = lr_model.predict(X_test_vectorized)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy:.3f} ({lr_accuracy*100:.1f}%)")

# Choose the best model
if lr_accuracy > nb_accuracy:
    best_model = lr_model
    best_predictions = lr_predictions
    best_model_name = "Logistic Regression"
    best_accuracy = lr_accuracy
else:
    best_model = nb_model
    best_predictions = nb_predictions
    best_model_name = "Naive Bayes"
    best_accuracy = nb_accuracy

print(f"\nBest model: {best_model_name} with {best_accuracy*100:.1f}% accuracy")

# Step 8: Detailed results
print("\n=== STEP 8: Detailed Results ===")
print("\nClassification Report:")
print(classification_report(y_test, best_predictions))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, 
            yticklabels=best_model.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Step 9: Function to classify new text
print("\n=== STEP 9: Using Your Model ===")

def classify_text(text):
    """
    Function to classify new text using your trained model
    """
    # Convert text to the same format as training data
    text_vectorized = vectorizer.transform([text])
    
    # Get prediction
    prediction = best_model.predict(text_vectorized)[0]
    
    # Get confidence scores
    probabilities = best_model.predict_proba(text_vectorized)[0]
    confidence = max(probabilities)
    
    return prediction, confidence

# Example usage
print("\nExample predictions:")
example_texts = [
    "Your example text here",  # Replace with relevant examples
    "Another example text",
    "Third example text"
]

for text in example_texts:
    if text.strip():  # Only if text is not empty
        prediction, confidence = classify_text(text)
        print(f"Text: '{text}'")
        print(f"Predicted category: {prediction}")
        print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        print("-" * 50)

# Step 10: Save your model (optional)
print("\n=== STEP 10: Saving Your Model ===")
import joblib

# Save the model and vectorizer
joblib.dump(best_model, 'text_classifier_model.pkl')
joblib.dump(vectorizer, 'text_vectorizer.pkl')
print("Model saved as 'text_classifier_model.pkl'")
print("Vectorizer saved as 'text_vectorizer.pkl'")

print("\n=== CONGRATULATIONS! ===")
print("You've successfully trained a text classification model!")
print(f"Your model achieved {best_accuracy*100:.1f}% accuracy")
print("\nTo use your model later, you can load it with:")
print("model = joblib.load('text_classifier_model.pkl')")
print("vectorizer = joblib.load('text_vectorizer.pkl')")

# Step 11: Model insights
print("\n=== STEP 11: Understanding What Your Model Learned ===")

if best_model_name == "Logistic Regression":
    # Show most important words for each category
    feature_names = vectorizer.get_feature_names_out()
    
    print("\nMost important words for each category:")
    for i, category in enumerate(best_model.classes_):
        print(f"\n{category}:")
        # Get coefficients for this category
        coefficients = best_model.coef_[i]
        # Get top 10 most important words
        top_indices = coefficients.argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"Key words: {', '.join(top_words)}")

print("\n" + "="*60)
print("NEXT STEPS:")
print("1. Replace 'your_file.csv' with your actual file name")
print("2. Update 'text_column' and 'label_column' with your column names") 
print("3. Run this code and see how well it works!")
print("4. Try the classify_text() function with new examples")
print("5. If accuracy is low, we can improve it together!")
print("="*60)