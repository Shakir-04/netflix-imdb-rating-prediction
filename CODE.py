import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

# Loading the CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Displaying the first few rows of the DataFrame
print("First few rows of the dataset:")
print(df.head())

# Displaying basic information about the DataFrame
print("\nBasic Information (df.info()):")
df.info()

# Displaying summary statistics for numerical columns
print("\nSummary Statistics (df.describe()):")
print(df.describe())

# Checking for missing values
print("\nMissing Values (df.isnull().sum()):")
print(df.isnull().sum())

# Displaying rows with missing values
print("\nRows with Missing Values:")
print(df[df.isnull().any(axis=1)])

# Filling missing values for categorical columns without using inplace=True
df['title'] = df['title'].fillna('Unknown Title')
df['genres'] = df['genres'].fillna('Unknown Genre')

# Filling missing values for numerical columns
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)


# Checking if missing values are filled
print("\nAfter filling missing values (df.isnull().sum()):")
print(df.isnull().sum())

# Replacing fields with `unknown ID`
df['imdbId'] = df['imdbId'].fillna('Unknown ID')
print(df.isnull().sum())

# Encoding categorical columns (example for 'genres')
df['genres'] = df['genres'].astype('category').cat.codes

# Splitting the dataset into training and testing sets
X = df.drop('imdbAverageRating', axis=1)  # Features
y = df['imdbAverageRating']  # Target

# Splitting into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Outputting the shapes of training and testing data
print("\nTraining and Testing Data Shapes:")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


##SECTION 2

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = "data.csv" 
df = pd.read_csv(url)

# Displaying the first few rows to understand the data structure
print(df.head())

# Checking the column names in your dataset
print(df.columns)

# Checking for missing values and handling them (dropping rows with missing target for simplicity)
df = df.dropna(subset=['imdbAverageRating']) 

## Checking for missing values
print("\nMissing Values (df.isnull().sum()):")
print(df.isnull().sum())

# Filling missing values for categorical columns
df['title'] = df['title'].fillna('Unknown Title')
df['genres'] = df['genres'].fillna('Unknown Genre')
df['availableCountries'] = df['availableCountries'].fillna('Unknown Country')
df['imdbId'] = df['imdbId'].fillna('Unknown IMDb ID')

# Filling missing values for numerical columns
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)

# Checking if missing values are filled
print("\nAfter filling missing values (df.isnull().sum()):")
print(df.isnull().sum())

# One-hot encoding for 'genres'
df = df.join(df['genres'].str.get_dummies(sep=', '))

# One-hot encoding for 'availableCountries'
df = df.join(df['availableCountries'].str.get_dummies(sep=', '))

# Dropping the original 'genres' and 'availableCountries' columns since we've one-hot encoded them
df.drop(columns=['genres', 'availableCountries','imdbId'], inplace=True)

# Encoding other categorical columns (example for 'type' using LabelEncoder)
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])  # You can apply LabelEncoder to other columns like 'type'

# Dropping any irrelevant features (e.g., 'title' if it's not needed for prediction)
df.drop(columns=['title'], inplace=True)

# Final check for data types to ensure everything is numeric
print("\nData types of features after encoding (df.dtypes):")
print(df.dtypes)


# Spliting the dataset into training and testing sets
X = df.drop('imdbAverageRating', axis=1)  # Features
y = df['imdbAverageRating']  # Target

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 1. Train Linear Regression Model

# Defining the Linear Regression model
linear_reg = LinearRegression()

# Training the model
linear_reg.fit(X_train, y_train)

# Predicting using the test set
y_pred_lr = linear_reg.predict(X_test)

# Evaluating the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Printing results for Linear Regression
print("Linear Regression Model Results:")
print(f"Mean Squared Error: {mse_lr}")
print(f"R-squared: {r2_lr}")

## Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model Training
rf_model.fit(X_train, y_train)

#Prediciton
y_pred_rf = rf_model.predict(X_test)

#Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

#output

print("Random Forest Regressor Results:")
print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Random Forest R-squared: {r2_rf}")


##SECTION 3

from sklearn.metrics import mean_absolute_error

# Generating Predictions
y_pred_lr = linear_reg.predict(X_test)  # Predictions from Linear Regression
y_pred_rf = rf_model.predict(X_test)   # Predictions from Random Forest Regressor

#Evaluating Metrics for Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mse_lr ** 0.5
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_lr}")
print(f"Root Mean Square Error (RMSE): {rmse_lr}")
print(f"R-squared (R²): {r2_lr}")

# Evaluation for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"Root Mean Square Error (RMSE): {rmse_rf}")
print(f"R-squared (R²): {r2_rf}")

# 4. Comparing Results
print("\nPerformance Comparison:")
print("Linear Regression:")
print(f"  - MAE: {mae_lr}, RMSE: {rmse_lr}, R²: {r2_lr}")
print("Random Forest Regressor:")
print(f"  - MAE: {mae_rf}, RMSE: {rmse_rf}, R²: {r2_rf}")

if r2_rf > r2_lr:
    print("\nThe Random Forest Regressor outperforms the Linear Regression model based on R².")
else:
    print("\nThe Linear Regression model performs better based on R².")

# Section 4
import matplotlib.pyplot as plt
import seaborn as sns

# Plot for Linear Regression
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.6, label='Linear Regression')
sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal Predictions')
plt.title("Linear Regression: Predicted vs. Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

# Plot for Random Forest
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6, label='Random Forest Regressor')
sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal Predictions')
plt.title("Random Forest Regressor: Predicted vs. Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()