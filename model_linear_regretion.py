

# Import libraries

import pandas as pd
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import my libraries
from utils import show_dataset_full

# Load the dataset
real_estate_data = pd.read_csv("D:/my_github_projects/a_simple_ml_model/Real_Estate.csv")

#--- Optional part 
# Show the dataset for user understanding
print(show_dataset_full(real_estate_data).to_string())
# OR 
# For Markdown format, please install tabulate
# "pip install tabulate"
# print(show_dataset_full(real_estate_data).to_markdown())
#--- Optional part end

# Selecting features and target variable
features = ["Distance to the nearest MRT station", "Number of convenience stores", "Latitude", "Longitude"]
target = "House price of unit area"

# Selecting the feature columns from the real estate dataset
X = real_estate_data[features]  

# Selecting the target variable (dependent variable) from the dataset
y = real_estate_data[target]  

# Splitting the dataset into training and testing sets
# test_size=0.2 (20% of the data will be used for testing, and 80% for training)
# random_state=42 (Ensures reproducibility by setting a fixed random seed for consistent results)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Why Use Linear Regression for the Model?
# ----------------------------------------
# 1. Simplicity and Interpretability
# 2. Assumption of Linearity
# 3. Efficiency
# 4. Baseline Model

# Model initialization
model = LinearRegression()

# Trainig the model
model.fit(X_train, y_train)

# Now visualize the actual versus predicted values to assess how well the model is performing.
# --------------------------------------------------------------------------------------------

# Making predictions using the Linear Regression Model
y_pred_linear_regression = model.predict(X_test)

# Plot a Graph for Visual Clearity
# --------------------------------

# Visualizing Actual vs. Predicted values
plot.figure(figsize=(10, 6))
plot.scatter(y_test, y_pred_linear_regression, alpha=0.5)
# Arguments to plot 'k--' - K for color 'black', '--' for dashed line style, 'lw' for line width , default width is 1
plot.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=3 )
plot.xlabel("Actual")
plot.ylabel("Predicted")
plot.title("Actual vs. Predicted House Prices")
plot.gcf().canvas.manager.set_window_title("Linear Regression Model: Prediction Accuracy")
plot.show()






