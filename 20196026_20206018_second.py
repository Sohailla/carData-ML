import pandas as pandas
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pandas.read_csv("customer_data.csv")

# Normalize the feature data
dataset["age"] = (dataset["age"] - dataset["age"].min()) / (dataset["age"].max() - dataset["age"].min())
dataset["salary"] = (dataset["salary"] - dataset["salary"].min()) / (dataset["salary"].max() - dataset["salary"].min())

# Scatter plot with different colors for the "purchased" feature
plot.scatter(dataset[dataset["purchased"]==0]["age"], dataset[dataset["purchased"]==0]["salary"], color='r', label='Not purchased')
plot.scatter(dataset[dataset["purchased"]==1]["age"], dataset[dataset["purchased"]==1]["salary"], color='b', label='Purchased')
plot.xlabel("Age")
plot.ylabel("Salary")
plot.legend()
plot.show()

# Split the dataset into training and testing sets
X = dataset[["age", "salary"]]
y = dataset["purchased"]
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
