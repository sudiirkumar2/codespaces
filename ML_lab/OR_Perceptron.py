from perceptron import Perceptron
import numpy as np

or_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
or_labels = np.array([0, 1, 1, 1])

print("\nOR Gate Features:")
print(or_features)
print("\nOR Gate Labels:")
print(or_labels)

# Instantiate a new Perceptron for the OR gate
perceptron_or = Perceptron(learning_rate=0.01, n_features=2)

# Train the perceptron with OR data
perceptron_or.train(or_features, or_labels, epochs=100)

# Test the trained OR model
print("\nTesting OR Gate Perceptron:")
for inputs, label in zip(or_features, or_labels):
    prediction = perceptron_or.predict(inputs)
    print(f"Input: {inputs}, Predicted: {prediction}, Actual: {label}")