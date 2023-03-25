import pandas as pandas
import matplotlib.pyplot as plot
import numpy as numpy
from sklearn.model_selection import train_test_split

# A. Load the “car_data.csv” dataset.
url = "car_data.csv"
carInfo = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
            'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
            'peakrpm', 'citympg', 'highwaympg', 'price']

dataset = pandas.read_csv(url)
# print(dataset.head())

# B. Use scatter plots between different features (7 at least) and the car price to select 5 of the numerical features
# that are positively/negatively correlated to the car price (i.e., features that have a relationship with the
# target). These 5 features are the features that will be used in linear regression.
# dataset.plot(kind='scatter', x='carlength', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='carwidth', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='carheight', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='curbweight', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='enginesize', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='boreratio', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='stroke', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='peakrpm', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='citympg', y='price')
# plot.show()
# dataset.plot(kind='scatter', x='highwaympg', y='price')
# plot.show()

# select 5 features that are correlated with price
features = ['carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower']

# C. Split the dataset into training and testing sets.
x = dataset[features]
y = dataset["price"]
# X_train: the training input features
# X_test: the testing input features
# y_train: the training target variable
# y_test: the testing target variable
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# D. Implement linear regression from scratch using gradient descent to
# optimize the parameters of the hypothesis function.

# define the hypothesis function
def hypothesis(X, theta):
    return numpy.dot(X, theta)

# define the cost function
def cost(X, Y, theta):
    m = len(X)
    h = hypothesis(X, theta)
    M_squared_error = (h - Y) **2
    calc_cost = 1/ (2*m) * numpy.sum(M_squared_error)
    return  calc_cost

# define the gradient descent function
# AND
# F. Calculate the cost (mean squared error) in every iteration to see how the
# error of the hypothesis function changes with every iteration of gradient
# descent.
def gradient_descent(X, Y, theta, alpha, n_iterations):
    m = len(X)
    result = []
    for i in range(n_iterations):
        h = hypothesis(X, theta)
        calc_errors = h - Y
        gradient = 1/ m * numpy.dot(X.T, calc_errors)
        theta = theta - alpha * gradient
        result.append(cost(X, Y, theta))
    return theta, result

# initialize theta and hypothesis parameters
alpha = 0.01
n_iterations = 1000
theta = numpy.zeros(X_train.shape[1])

# run gradient descent to optimize theta
theta, result = gradient_descent(X_train, y_train, theta, alpha, n_iterations)

# E. print the optimized parameters
print('Optimized parameters:')
print(theta)

# G. Plot the cost against the number of iterations.
plot.plot(result)
plot.xlabel('Iterations')
plot.ylabel('Cost')
plot.show()

# H. Use the optimized hypothesis function to make predictions on the testing
# set and calculate the accuracy of the final (trained) model on the test set.
h = hypothesis(theta, X_test)
Accuracy = 1 - numpy.mean(numpy.abs(h - y_test)/y_test)
print('Accuracy:', Accuracy)