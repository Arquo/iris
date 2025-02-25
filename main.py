import math
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class LogisticRegressionScratchMultiClass:
    # learning rate: step size the model updates
    # epochs: number of time iterate through the entire dataset
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.biases = []

    def sigmoid(self, z):
        # to map any number to 0 to 1
        return 1 / (1 + math.exp(-z))

    def train_binary(self, X, y):
        n_features = len(X[0])
        weights = [0] * n_features
        bias = 0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(weights, xi)) + bias
                y_pred = self.sigmoid(z)
                error = y_pred - yi
                weights = [w - self.lr * error * x for w, x in zip(weights, xi)]
                bias -= self.lr * error
        return weights, bias

    def train(self, X, y):
        self.classes = sorted(set(y))
        for c in self.classes:
            binary_y = [1 if label == c else 0 for label in y]
            w, b = self.train_binary(X, binary_y)
            self.weights.append(w)
            self.biases.append(b)

    def predict_proba(self, x):
        probs = []
        for w, b in zip(self.weights, self.biases):
            z = sum(wi * xi for wi, xi in zip(w, x)) + b
            probs.append(self.sigmoid(z))
        return probs

    def predict(self, X):
        return [int(self.classes[max(range(len(self.classes)), key=lambda i: self.predict_proba(x)[i])]) for x in X]


# Load and preprocess dataset
iris_csv = pd.read_csv("C:\\Users\\RAY\\Desktop\\Python\\IrisFlowerClassification\\Iris.csv")
iris_csv.drop(columns=['Id'], inplace=True)
X = iris_csv.drop(columns=['Species']).values.tolist()
y = LabelEncoder().fit_transform(iris_csv['Species'])  # 0, 1, 2 for three classes

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= random.randint(1, 1000))

# Train and evaluate
model = LogisticRegressionScratchMultiClass(lr=0.1, epochs=1000)
model.train(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = sum(yt == yp for yt, yp in zip(y_test, y_pred)) / len(y_test)
print("Actual:", y_test)
print("Predicted:", y_pred)
print(f"Accuracy: {accuracy:.4f}")