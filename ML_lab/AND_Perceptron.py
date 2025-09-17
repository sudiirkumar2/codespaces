import numpy as np
from perceptron import Perceptron

and_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_labels = np.array([0, 0, 0, 1])

print("AND Gate Features:")
print(and_features)
print("\nAND Gate Labels:")
print(and_labels)

# Instantiate the Perceptron
perceptron_and = Perceptron(learning_rate=0.3, n_features=2)

# Train the perceptron with AND data
perceptron_and.train(and_features, and_labels, epochs=100)

# Test the trained model
print("Testing AND Gate Perceptron:")
for inputs, label in zip(and_features, and_labels):
    prediction = perceptron_and.predict(inputs)
    print(f"Input: {inputs}, Predicted: {prediction}, Actual: {label}")