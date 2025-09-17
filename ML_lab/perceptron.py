import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.3, n_features=None):
        if not n_features:
            raise ValueError("n_features must be provided during initialization.")
        self.learning_rate = learning_rate
        self.weights = np.random.rand(n_features)
        self.bias = 0.4

    def predict(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        return 1 if linear_output >= 0 else 0

    def train(self, training_features, training_labels, epochs):
        for _ in range(epochs):
            tot_err = 0
            print('Current weights:',self.weights)
            for inputs, label in zip(training_features, training_labels):
                prediction = self.predict(inputs)
                update = self.learning_rate * (label - prediction)
                tot_err += abs(label - prediction)
                self.weights += update * inputs
                self.bias += update
            print(f'Epoch: {_}, Total error: {tot_err}')
            if tot_err==0:
                break
            