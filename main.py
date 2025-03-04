import math
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime




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
        n_features = len(X[0])  # number of class in training X
        weights = [0] * n_features   # initialize the weight
        bias = 0 # initialize 0

        for _ in range(self.epochs):  # go through the loop for #self.epochs times, _ is a plaseholder, we dont use it
            for xi, yi in zip(X, y): # pairs of X and y, now y is 1 and 0, now looking at only a pair of X and y
                z = sum(w * x for w, x in zip(weights, xi)) + bias  # a score,
                y_pred = self.sigmoid(z) # map it to [0,1]
                error = y_pred - yi # difference of mapped and real y,   Gradient for bias
                weights = [w - self.lr * error * x for w, x in zip(weights, xi)] # adjust the weight base on this iteration
                                                                                 #[w1w2w3w4] - lr * error *[x1x2x3x4]
                bias -= self.lr * error
        return weights, bias

    def train(self, X, y):
        # x train, y train data
        self.classes = sorted(set(y)) # find all unique y, sort to 0,1,2, save them in classes
        for c in self.classes: # iterate through 0,1,2
            binary_y = [1 if label == c else 0 for label in y] # categorize them into A or Not A 1, and 0
            w, b = self.train_binary(X, binary_y)   # get the weight and bias.
            self.weights.append(w)
            self.biases.append(b)      # save the weight and bias.

    def predict_proba(self, x): # take a X testing data
        probs = []
        for w, b in zip(self.weights, self.biases):
            z = sum(wi * xi for wi, xi in zip(w, x)) + b # calculate the score for each class
            probs.append(self.sigmoid(z))  # convert and append the probability to a list
        return probs # return the list

    def predict(self, X):
        return [int(self.classes[max(range(len(self.classes)), key=lambda i: self.predict_proba(x)[i])]) for x in X]
    # return the max probability
    #

# Get the current timestamp
timestamp = datetime.now().isoformat()

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