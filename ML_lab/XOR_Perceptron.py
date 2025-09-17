import numpy as np
from perceptron import Perceptron

xor_features = np.array([[0,0],[0,1],[1,0],[1,1]])
xor_labels = np.array([0,1,1,0])

print("\nXOR Gate Features:")
print(xor_features)
print("\nXOR Gate Labels:")
print(xor_labels)

# Instantiate a new Perceptron for the OR gate
perceptron_xor = Perceptron(learning_rate=0.01, n_features=2)

# Train the perceptron with OR data
perceptron_xor.train(xor_features, xor_labels, epochs=100)

# Test the trained OR model
print("\nTesting XOR Gate Perceptron:")
for inputs, label in zip(xor_features, xor_labels):
    prediction = perceptron_xor.predict(inputs)
    print(f"Input: {inputs}, Predicted: {prediction}, Actual: {label}")