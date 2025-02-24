# Traditional Machine Learning:
#
# Logistic Regression
# k-Nearest Neighbors (k-NN)
# Decision Trees
# Random Forest
# Support Vector Machine (SVM)
# Deep Learning:
#
# Neural Networks (e.g., TensorFlow/Keras, PyTorch)
# Convolutional Neural Networks (if using images)
# Statistical Methods:
#
# Bayesian Classifiers
# Linear Discriminant Analysis (LDA)
# Unconventional Methods:
#
# Rule-based classification
# Genetic algorithms
# Fuzzy logic

import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
iris_csv = pd.read_csv("C:\\Users\\RAY\\Desktop\\Python\\IrisFlowerClassification\\Iris.csv")


iris_csv.drop(columns=['Id'], inplace=True)
X = iris_csv.drop(columns=['Species'])
y = iris_csv['Species']


# Encode target labels (converts categorical species names into numbers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Setosa -> 0, Versicolor -> 1, Virginica -> 2

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1, 1000))

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
print(y_test)
print(y_pred)
print("Accuracy: ",(1-sum(abs(y_test-y_pred))/len(y_test)).round(4))

# Evaluate model
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# # Predict a new flower sample (example)
# new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example Sepal/Petal measurements
# prediction = model.predict(new_sample)
# predicted_species = label_encoder.inverse_transform(prediction)
# print(f"Predicted Species: {predicted_species[0]}")








