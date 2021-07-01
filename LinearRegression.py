import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, iterations=1000):
        self.iterations = iterations
        self.lr = lr
        self.weights = None
        self.bias = None

    # number of weights == number of features, bias == 0
    def fit(self, X, y):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for _ in range(self.iterations):
            # y = mx + b
            y_predict = np.dot(X, self.weights) + self.bias

            # derivation
            dw = (1 / samples) * np.dot(X.T, (y_predict - y))
            db = (1 / samples) * np.sum(y_predict - y)

            # new weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predict = np.dot(X, self.weights) + self.bias

        return y_predict
