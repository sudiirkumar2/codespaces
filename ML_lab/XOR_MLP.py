from sklearn.neural_network import MLPClassifier
import numpy as np

# XOR input and output
data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 1, 1, 0])


# MLPClassifier with 4 hidden neurons and 'adam' solver for better accuracy
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', solver='adam', max_iter=10000, random_state=42, verbose=True)
mlp.fit(data, labels)

print("\nXOR MLP Results:")
for x, y in zip(data, labels):
    pred = mlp.predict([x])[0]
    print(f"Input: {x}, Predicted: {pred}, Actual: {y}")
