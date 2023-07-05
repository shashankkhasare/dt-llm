import numpy as np
from sklearn.tree import DecisionTreeClassifier
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Preprocessing and preparing the text data
# Load the dataset from Hugging Face
dataset = load_dataset("imdb")

# Extract the text and label columns from the dataset
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

# Initialize the LLM and tokenizer
model_name = 'bert-base-uncased'  # Replace with your preferred LLM model
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the texts
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

# Generate importance scores using LLM
input_ids = tokenized_texts['input_ids']
attention_masks = tokenized_texts['attention_mask']
outputs = model(input_ids, attention_mask=attention_masks)
importance_scores = outputs[0].numpy()

# Select top N important features
N = 10  # Number of top features to select
top_features_indices = np.argsort(-importance_scores, axis=1)[:, :N]
selected_features = [tokenizer.convert_ids_to_tokens(indices) for indices in top_features_indices]

# Convert the selected features to a feature matrix
feature_matrix = np.zeros((len(texts), N))
for i, features in enumerate(selected_features):
    for j, feature in enumerate(features):
        feature_matrix[i, j] = texts[i].count(feature)

# Build the decision tree
tree = DecisionTreeClassifier()
tree.fit(feature_matrix, labels)

# Use the decision tree for prediction
new_texts = [
    "The movie was fantastic!",
    "I would not recommend this product.",
    "The service was excellent.",
    "The book was disappointing."
]
tokenized_new_texts = tokenizer(new_texts, padding=True, truncation=True, return_tensors='tf')
new_input_ids = tokenized_new_texts['input_ids']
new_attention_masks = tokenized_new_texts['attention_mask']
new_outputs = model(new_input_ids, attention_mask=new_attention_masks)
new_importance_scores = new_outputs[0].numpy()

new_feature_matrix = np.zeros((len(new_texts), N))
for i, features in enumerate(selected_features):
    for j, feature in enumerate(features):
        if i < len(new_texts) and feature in new_texts[i]:
            new_feature_matrix[i, j] = new_texts[i].count(feature)

predictions = tree.predict(new_feature_matrix)
print(predictions)