import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("customer_data.csv")

# Normalize the feature data
data["age"] = (data["age"] - data["age"].min()) / (data["age"].max() - data["age"].min())
data["salary"] = (data["salary"] - data["salary"].min()) / (data["salary"].max() - data["salary"].min())

# Scatter plot with different colors for the "purchased" feature
plt.scatter(data[data["purchased"]==0]["age"], data[data["purchased"]==0]["salary"], color='r', label='Not purchased')
plt.scatter(data[data["purchased"]==1]["age"], data[data["purchased"]==1]["salary"], color='b', label='Purchased')
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Split the dataset into training and testing sets
X = data[["age", "salary"]]
y = data["purchased"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run logistic regression and optimize the parameters
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Print the parameters of the hypothesis function
print("Intercept:", clf.intercept_)
print("Coefficients:", clf.coef_)

# Use the optimized hypothesis function to make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the final (trained) model on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy = ", accuracy)
