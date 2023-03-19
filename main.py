import pandas as pandas
from matplotlib import pyplot as plt
import numpy as np

# # A. Load the “car_data.csv” dataset.
# url = "/Users/macintoshhd/Downloads/Sohaila /Final-Second-Term/Machine Learning/ass1/car_data.csv"
# carInfo = ['symboling', 'name', 'fueltypes', 'doornumbers', 'carbody', 'drivewheels', 'enginelocation', 'wheelbase',
#          'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'fuelsystem',
#          'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peak rpm', 'citympg', 'highwaympg', 'price']
# dataset = pandas.read_csv(url, names=carInfo)
# print(dataset.head())
# load the dataset
dataset = pandas.read_csv('/Users/macintoshhd/Downloads/Sohaila /Final-Second-Term/Machine Learning/ass1/car_data.csv')

carInfo = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
            'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
            'peakrpm', 'citympg', 'highwaympg', 'price']

# B. Use scatter plots between different features (7 at least) and the car price to select 5 of the numerical features
# that are positively/negatively correlated to the car price (i.e., features that have a relationship with the
# target). These 5 features are the features that will be used in linear regression.
for car in carInfo:
    plt.scatter(dataset[car], dataset['price'])
    plt.xlabel(car)
    plt.ylabel('Price')
    plt.show()

# select 5 features that are correlated with price
carInfo = ['carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower']

# C. Split the dataset into training and testing sets.
size = int(0.8 * len(dataset))
data = dataset.sample(size, random_state= 42)
test = dataset.drop(data.index)

# X = dataset[['enginesize', 'horsepower', 'carwidth', 'curbweight', 'citympg']]
# Y = dataset['price']
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=2, random_state=42)

# D. Implement linear regression from scratch using gradient descent to
# optimize the parameters of the hypothesis function.
# define the hypothesis function
def hypothesis(theta, X):
    return np.dot(X, theta)

# define the cost function
def cost(theta, X, y):
    m = len(y)
    h = hypothesis(theta, X)
    J = np.sum((h-y)**2)/(2*m)
    return J

# define the gradient descent function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        h = hypothesis(theta, X)
        theta = theta - (alpha/m)*np.dot(X.T, (h-y))
        cost_history[i] = cost(theta, X, y)
    return theta, cost_history

# prepare training and testing data
train_X = np.array(data[carInfo])
train_y = np.array(data['price'])
test_X = np.array(test[carInfo])
test_y = np.array(test['price'])

# normalize training data
mu = np.mean(train_X, axis=0)
sigma = np.std(train_X, axis=0)
train_X = (train_X - mu)/sigma

# add bias term to training data
train_X = np.insert(train_X, 0, 1, axis=1)

# initialize theta and hyperparameters
theta = np.zeros(train_X.shape[1])
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
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# normalize testing data
test_X = (test_X - mu)/sigma

# add bias term to testing data
test_X = np.insert(test_X, 0, 1, axis=1)

# H. Use the optimized hypothesis function to make predictions on the testing
# set and calculate the accuracy of the final (trained) model on the test set.
predictions = hypothesis(theta, test_X)
accuracy = 1 - np.mean(np.abs(predictions - test_y)/test_y)
print('Accuracy:', accuracy)