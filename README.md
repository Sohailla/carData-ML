<div align="center">
  <h2>Machine Learning</h2>
  <h3>Assingment 1 - linear and logistic regression</h3>
  <hr />
</div>

- A certain car company is planning to manufacture and launch a new car. So, the company’s consultants need to study the factors which the pricing of cars depends on. Based on various market surveys, the consultants gathered a dataset of different types of cars across the market. They would like to know which of the car features are significant in predicting the price of a car.

- In addition, they would like to predict whether customers will be interested in purchasing the new car. That’s why they collected a few records of some of the company’s previous customers who either purchased a new car from the company as an upgrade or didn’t purchase a new car.

- You are required to build a linear regression model and a logistic regression model for this company to predict car prices and purchases based on some features.

## 1. Data:

There are 2 attached datasets:

- The first dataset “car_data.csv” contains 205 records of cars with 25
  features per record in addition to 1 target column. These features include the car size and dimensions, the fuel system and fuel type used by the car, the engine size and type, the horsepower, etc. The final column (i.e., the target) is the car price (in some monetary unit).

- The second dataset “customer_data.csv” contains 400 records representing some of the company’s previous customers. The customer data is composed of the customer’s age and salary. The final column (i.e., the target) is a boolean value (0 if the customer didn’t purchase a new car and 1 if he/she purchased a new car).

## 2. Requirements:

Write 2 python programs (2 separated .py files) in which you work on each dataset (each model) separately as follows:

- **In the first program:**

  1.  Load the “car_data.csv” dataset.
  2.  Use scatter plots between different features (7 at least) and the car price
      to select 5 of the numerical features that are positively/negatively correlated to the car price (i.e., features that have a relationship with the target). These 5 features are the features that will be used in linear regression.
  3.  Split the dataset into training and testing sets.
  4.  Implement linear regression from scratch using gradient descent to
      optimize the parameters of the hypothesis function.
  5.  Print the parameters of the hypothesis function.
  6.  Calculate the cost (mean squared error) in every iteration to see how the
      error of the hypothesis function changes with every iteration of gradient
      descent.
  7.  Plot the cost against the number of iterations.
  8.  Use the optimized hypothesis function to make predictions on the testing
      set and calculate the accuracy of the final (trained) model on the test set.

- **In the second program:**
  1. Load the “customer_data.csv” dataset.
  2. Normalize the feature data (the customer’s age and salary) before
     applying regression. You can use minmax normalization where z is the
     normalized value and z = (x – min) / (max – min).
  3. Use scatter plot between the customer’s age and salary and differentiate
     between the purchased by colors (ex: red for y=0 and blue for y=1)
  4. Split the dataset into training and testing sets.
  5. Run logistic regression (from sklearn) to optimize the parameters of the
     hypothesis function. Use the 2 features (age & salary) as input and the output to be predicted is “purchased” (Do not implement the logistic regression from scratch).
  6. Print the parameters of the hypothesis function.
  7. Use the optimized hypothesis function to make predictions on the testing
     set.
  8. Calculate the accuracy of the final (trained) model on the test set.

## 3. Grading Criteria

### 1. First Program (11 Marks)

| Criteria                            | Mark |
| ----------------------------------- | :--: |
| Load the dataset/Splitting the data |  1   |
| Scatter plots for feature selection |  1   |
| Linear regression                   |  6   |
| MSE (calculation and plot)          |  6   |
| Test the hypothesis function        |  6   |

### 2. Second Program (7 Marks)

| Criteria                            | Mark |
| ----------------------------------- | :--: |
| Load the dataset/Splitting the data |  1   |
| Scatter plots                       |  1   |
| Normalization                       |  1   |
| Logistic regression                 |  3   |
| Predictions on testing set/Accuracy |  1   |

18/3 = 6 Marks
