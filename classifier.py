import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import json

# Preprocessing and preparing the text data
# Load the dataset from Hugging Face
dataset = load_dataset("imdb")

# Extract the text and label columns from the dataset
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

# Create custom backend and pass it to KeyBERT
model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cuda:1")
keybert_model = KeyBERT(model)
selected_features = set()
for text in texts:
    keywords = keybert_model.extract_keywords(text)
    selected_features.update([keyword[0] for keyword in keywords])

# Convert the selected features to a feature matrix
N = len(selected_features)
feature_matrix = np.zeros((len(texts), N))
for i in range(len(texts)):
    for j, feature in enumerate(selected_features):
        feature_matrix[i, j] = texts[i].count(feature)

# Build the decision tree
tree = DecisionTreeClassifier()
tree.fit(feature_matrix, labels)

# Print the decision tree
tree_text = export_text(tree, feature_names=list(selected_features))
print(tree_text)

# Use the decision tree for prediction
new_texts = dataset["test"]["text"]

new_feature_matrix = np.zeros((len(new_texts), N))
for i in range(len(new_texts)):
    for j, feature in enumerate(selected_features):
        new_feature_matrix[i, j] = new_texts[i].count(feature)

predictions = tree.predict(new_feature_matrix)
accuracy = accuracy_score(dataset["test"]["label"], predictions)
print(f"Accuracy: {accuracy}")