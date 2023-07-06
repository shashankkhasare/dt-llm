import json
import numpy as np
import pickle
from sklearn.tree import export_text

tree = pickle.load(open('tree.model', 'rb'))
selected_features = pickle.load(open('keywords.list', 'rb'))

# Print the decision tree
tree_text = export_text(tree, feature_names=list(selected_features))
print(tree_text)


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
text_file.write(tree_json)
text_file.close()