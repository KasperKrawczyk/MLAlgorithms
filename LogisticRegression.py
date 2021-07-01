import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for _ in range(self.iterations):
            logistic_model = np.dot(X, self.weights) + self.bias
            y_predict = self.sig(logistic_model)

            # derivation
            dw = (1 / samples) * np.dot(X.T, (y_predict - y))
            db = (1 / samples) * np.sum(y_predict - y)

            # new weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        logistic_model = np.dot(X, self.weights) + self.bias
        y_predict = self.sig(logistic_model)

        y_predict_class = []

        for i in y_predict:
            if i > 0.5:
                y_predict_class.append(1)
            else:
                y_predict_class.append(0)

        return y_predict_class

    def sig(self, X):
        sigmoid = 1 / (1 + np.exp(-X))

        return sigmoid