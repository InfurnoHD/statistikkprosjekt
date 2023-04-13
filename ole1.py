import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Read the combined data
combined_data = pd.read_csv('combined_data.csv', parse_dates=['Time'])

# Convert time to Unix time (seconds since the epoch)
combined_data['Time'] = combined_data['Time'].apply(lambda dt: dt.timestamp())

# Convert Unix time to hour of the day
combined_data['Hour'] = combined_data['Time'].apply(lambda ts: datetime.fromtimestamp(ts).hour)

# Convert timestamp to season
def get_season(timestamp):
    month = timestamp.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

combined_data['Season'] = combined_data['Time'].apply(lambda ts: get_season(datetime.fromtimestamp(ts)))

# Extract the features and target variable
X = combined_data['Consumption'].values.reshape(-1, 1)
y_hour = combined_data['Hour'].values
y_temperature = combined_data['Temperature'].values

# Split the data into training and testing sets
X_train, X_test, y_hour_train, y_hour_test = train_test_split(X, y_hour, test_size=0.2, random_state=42)
_, _, y_temperature_train, y_temperature_test = train_test_split(X, y_temperature, test_size=0.2, random_state=42)

# Create a linear regression model for temperature
temperature_model = LinearRegression()

# Train the temperature model using the training data
temperature_model.fit(X_train, y_temperature_train)

# Make predictions using the testing data
y_temperature_pred = temperature_model.predict(X_test)

# Create a polynomial regression model for hour of the day
polynomial_features = PolynomialFeatures(degree=3)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.transform(X_test)

hour_model = LinearRegression()

# Train the hour of the day model using the training data
hour_model.fit(X_train_poly, y_hour_train)

# Make predictions using the testing data
y_hour_pred = hour_model.predict(X_test_poly)

# Group the data by season
seasons = combined_data.groupby('Season')

# Plot the actual vs. predicted values for hour of the day for each season
for season, data in seasons:
    X_season = data['Consumption'].values.reshape(-1, 1)
    y_hour_season = data['Hour'].values
    X_season_poly = polynomial_features.transform(X_season)
    y_hour_season_pred = hour_model.predict(X_season_poly)
    plt.scatter(X_season, y_hour_season, color='blue', label='Actual')
    plt.scatter(X_season, y_hour_season_pred, color='red', label='Predicted')
    plt.xlabel('Consumption')
    plt.ylabel('Hour of the Day')
    plt.ylim(0, 23)
    plt.legend()
    plt.title(f'Actual vs Predicted Hour of the Day ({season})')
    plt.show()

# Plot the actual vs. predicted values for temperature
plt.scatter(X_test, y_temperature_test, color='blue', label='Actual')
plt.scatter(X_test, y_temperature_pred, color='red', label='Predicted')
plt.xlabel('Consumption')
plt.ylabel('Temperature')
plt.legend()
plt.title('Actual vs Predicted Temperature (Linear Regression)')
plt.show()

# Calculate the mean squared error and R-squared value for each model
mse_hour = mean_squared_error(y_hour_test, y_hour_pred)
r2_hour = r2_score(y_hour_test, y_hour_pred)
mse_temperature = mean_squared_error(y_temperature_test, y_temperature_pred)
r2_temperature = r2_score(y_temperature_test, y_temperature_pred)

print(f'Mean Squared Error (Hour of the Day): {mse_hour}')
print(f'R-squared (Hour of the Day): {r2_hour}')
print(f'Mean Squared Error (Temperature): {mse_temperature}')
print(f'R-squared (Temperature): {r2_temperature}')

# Save the model coefficients
coefficients = pd.DataFrame({'Feature': ['Consumption'],
                             'Hour_Coefficient': [hour_model.coef_[0]],
                             'Temperature_Coefficient': [temperature_model.coef_[0]]})
coefficients.to_csv('model_coefficients.csv', index=False)

print("Model coefficients saved in 'model_coefficients.csv'")
