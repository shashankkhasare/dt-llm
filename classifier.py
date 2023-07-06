import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import json
import pickle

# Preprocessing and preparing the text data
# Load the dataset from Hugging Face
dataset = load_dataset("imdb")

# Extract the text and label columns from the dataset
texts = dataset["train"]["text"][0:20000]
labels = dataset["train"]["label"][0:20000]

# Create custom backend and pass it to KeyBERT
model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cuda:1")
keybert_model = KeyBERT(model)
selected_features = set()
for i, text in enumerate(texts):
    print(f"review number: {i}")
    keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    selected_features.update([keyword[0] for keyword in keywords])
selected_features = list(selected_features)

# Convert the selected features to a feature matrix
N = len(selected_features)
feature_matrix = np.zeros((len(texts), N))
for i in range(len(texts)):
    for j, feature in enumerate(selected_features):
        feature_matrix[i, j] = texts[i].count(feature)

# Build the decision tree
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(feature_matrix, labels)
pickle.dump(tree, open('tree.model', 'wb'))
keywordlist = list(selected_features)
pickle.dump(keywordlist, open('keywords.list', 'wb'))

# Print the decision tree
tree_text = export_text(tree, feature_names=list(selected_features))
pickle.dump(tree_text, open('tree_text.diag', 'wb'))

# Convert the decision tree to JSON-like structure
def tree_to_dict(tree, feature_names, node_index=0):
    tree_dict = {}
    if node_index == -1 or tree.children_left[0] == -1:  # Leaf node
        class_counts = tree.value[node_index].flatten()
        majority_class_index = np.argmax(class_counts)
        tree_dict['class'] = int(majority_class_index)
    else:  # Internal node
        feature_index = tree.feature[node_index]
        threshold = tree.threshold[node_index]
        feature_name = feature_names[feature_index]
        tree_dict['feature'] = feature_name
        tree_dict['threshold'] = float(threshold)
        left_child_index = tree.children_left[node_index]
        right_child_index = tree.children_right[node_index]
        tree_dict['left'] = tree_to_dict(tree, feature_names, left_child_index)
        tree_dict['right'] = tree_to_dict(tree, feature_names, right_child_index)
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