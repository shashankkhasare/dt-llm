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
for i, text in enumerate(texts):
    print(f"review number: {i}")
    keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
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

# Convert the decision tree to JSON-like structure
def tree_to_dict(tree, feature_names):
    tree_dict = {}
    if tree.children_left[0] == -1:  # Leaf node
        tree_dict['class'] = int(tree.value[0, 0])
    else:  # Internal node
        feature_index = tree.feature[0]
        threshold = tree.threshold[0]
        feature_name = feature_names[feature_index]
        tree_dict['feature'] = feature_name
        tree_dict['threshold'] = float(threshold)
        tree_dict['left'] = tree_to_dict(tree.tree_.children_left[0], feature_names)
        tree_dict['right'] = tree_to_dict(tree.tree_.children_right[0], feature_names)
    return tree_dict

tree_dict = tree_to_dict(tree.tree_, selected_features)
tree_json = json.dumps(tree_dict, indent=4)
text_file = open("tree.json", "w")
n = text_file.write(tree_json)
text_file.close()
print(tree_json)

# Use the decision tree for prediction
new_texts = dataset["test"]["text"]

new_feature_matrix = np.zeros((len(new_texts), N))
for i in range(len(new_texts)):
    for j, feature in enumerate(selected_features):
        new_feature_matrix[i, j] = new_texts[i].count(feature)

predictions = tree.predict(new_feature_matrix)
accuracy = accuracy_score(dataset["test"]["label"], predictions)
print(f"Accuracy: {accuracy}")