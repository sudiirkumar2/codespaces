import pandas as pd
from collections import Counter
import math
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

data = pd.read_csv('personalized_learning_dataset.csv')

def entropy(target_col):
    counts=Counter(target_col)
    # print(counts)
    total=len(target_col)

    ent=0
    for label in counts:
        prob=counts[label]/total
        ent-=prob*math.log2(prob)
    return ent        

def InformationGain(data,feature,target_name="play"):
    total_entropy=entropy(data[target_name]) #D
    values=data[feature].unique() #unique values in a column

    weighted_entropy=0
    for value in values: #iterating over all the possible values of a column
        subset=data[data[feature]==value]
        # print(subset)
        weight=len(subset)/len(data)
        weighted_entropy+=weight*entropy(subset[target_name])

    gain=total_entropy-weighted_entropy
    return gain

def ID3(data,full_data,features,target="play",default_class=None):
    labels=data[target].unique()

    if len(labels)==1:
        return labels[0]

    if len(data)==0:
        majority_label=full_data[target].mode()[0]
        return majority_label

    if len(features)==0:
        majority_label=data[target].mode()[0]
        return majority_label

    gains=[InformationGain(data,feature,target) for feature in features]
    best_index=gains.index(max(gains))
    best_feature=features[best_index]

    # print(gains)

    tree={best_feature:{}}

    remaining_features=[f for f in features if f!=best_feature]


    for value in data[best_feature].unique():
        subset=data[data[best_feature]==value]
        subtree=ID3(subset,full_data,remaining_features,target)
        tree[best_feature][value]=subtree

    return tree


# Select categorical features for ID3
features = ['Gender', 'Education_Level', 'Course_Name', 'Learning_Style']
target = 'Dropout_Likelihood'

# Convert categorical columns to string (if not already)
for col in features + [target]:
    data[col] = data[col].astype(str)

tree = ID3(data, data, features, target=target)
print(tree)


# Function to predict using the built tree
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    feature_value = sample[feature]
    subtree = tree[feature].get(feature_value)
    if subtree is None:
        # If value not seen in training, return None or majority class
        return None
    return predict(subtree, sample)


# Load test data
test_data = pd.read_csv('test.csv')


# Debug: print test_data columns
print('Test columns:', test_data.columns.tolist())

# Ensure test data uses the same categorical features
for col in features:
    if col in test_data.columns:
        test_data[col] = test_data[col].astype(str)
    else:
        print(f"Warning: Column '{col}' not found in test data.")



# If test data has actual labels, compare predictions
if target in test_data.columns:
    y_true = test_data[target].tolist()
    X_test = test_data.drop(target, axis=1)
else:
    y_true = None
    X_test = test_data


results = []
for _, row in X_test.iterrows():
    sample = row.to_dict()
    label = predict(tree, sample)
    results.append(label)
print('Predictions:', results)


# If actual labels are available, print analysis metrics
if y_true is not None:
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, results, labels=['Yes', 'No']))
    print('Accuracy:', accuracy_score(y_true, results))
    print('Precision:', precision_score(y_true, results, pos_label='Yes'))
    print('Recall:', recall_score(y_true, results, pos_label='Yes'))