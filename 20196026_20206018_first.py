import pandas as pandas
import matplotlib.pyplot as plot
import numpy as numpy
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

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
dataset.plot(kind='scatter', x='carlength', y='price')
plot.show()
dataset.plot(kind='scatter', x='carwidth', y='price')
plot.show()
dataset.plot(kind='scatter', x='carheight', y='price')
plot.show()
dataset.plot(kind='scatter', x='curbweight', y='price')
plot.show()
dataset.plot(kind='scatter', x='enginesize', y='price')
plot.show()
dataset.plot(kind='scatter', x='boreratio', y='price')
plot.show()
dataset.plot(kind='scatter', x='stroke', y='price')
plot.show()
dataset.plot(kind='scatter', x='peakrpm', y='price')
plot.show()
dataset.plot(kind='scatter', x='citympg', y='price')
plot.show()
dataset.plot(kind='scatter', x='highwaympg', y='price')
plot.show()

# select 5 features that are correlated with price
carInfo = ['carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower']

# C. Split the dataset into training and testing sets.
# data = dataset.sample(frac=0.8, random_state=42)  // pandas
# test = dataset.drop(data.index)
data, test = train_test_split(dataset, test_size=0.8, random_state=42)

# D. Implement linear regression from scratch using gradient descent to
# optimize the parameters of the hypothesis function.
# define the hypothesis function
def hypothesis(Theta, X):
    return numpy.dot(X, Theta)

# define the cost function
def cost(Theta, X, y):
    m = len(y)
    h = hypothesis(theta, X)
    J = numpy.sum((h-y)**2)/(2*m)
    return J

# define the gradient descent function
def gradient_descent(X, y, Theta, Alpha, num_iterations):
    m = len(y)
    costCycle = numpy.zeros(num_iterations)
    for i in range(iterations):
        h = hypothesis(Theta, X)
        Theta = Theta - (Alpha / m) * numpy.dot(X.T, (h - y))
        costCycle[i] = cost(theta, X, y)
    return Theta, costCycle

# prepare training and testing data
train_X = numpy.array(data[carInfo])
train_y = numpy.array(data['price'])
test_X = numpy.array(test[carInfo])
test_y = numpy.array(test['price'])

# normalize training data
mu = numpy.mean(train_X, axis=0)
sigma = numpy.std(train_X, axis=0)
train_X = (train_X - mu)/sigma

# add bias term to training data
train_X = numpy.insert(train_X, 0, 1, axis=1)

# initialize theta and hyperparameters
theta = numpy.zeros(train_X.shape[1])
alpha = 0.01
iterations = 1500

# run gradient descent to optimize theta
theta, cost_history = gradient_descent(train_X, train_y, theta, alpha, iterations)

# print the optimized parameters
print('Optimized parameters:')
print(theta)
# F. Calculate the cost (mean squared error) in every iteration to see how the
# error of the hypothesis function changes with every iteration of gradient
# descent.

# G. Plot the cost against the number of iterations.
plot.plot(cost_history)
plot.xlabel('Iterations')
plot.ylabel('Cost')
plot.show()

# normalize testing data
test_X = (test_X - mu)/sigma

# add bias term to testing data
test_X = numpy.insert(test_X, 0, 1, axis=1)

# H. Use the optimized hypothesis function to make predictions on the testing
# set and calculate the accuracy of the final (trained) model on the test set.
predictions = hypothesis(theta, test_X)
accuracy = 1 - numpy.mean(numpy.abs(predictions - test_y)/test_y)
print('Accuracy:', accuracy)