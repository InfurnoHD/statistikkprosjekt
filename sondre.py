import pandas as pd
from datetime import datetime

# Function to parse the date and time
def parse_datetime(date_string):
    for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%d.%m.%YT%H:%M:%S.%fZ']:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse date string: {date_string}")

# Read temperature data
temperature_data = pd.read_csv('weather_average.csv', delimiter=';', parse_dates=['referenceTime'], date_parser=parse_datetime)

# Rename columns for clarity
temperature_data.columns = ['Time', 'Temperature']

# Read power consumption data
power_data = pd.read_csv('power.csv', delimiter=',', parse_dates=['referenceTime'], date_parser=parse_datetime)

# Rename columns for clarity
power_data.columns = ['Time', 'Consumption']

# Merge temperature and power consumption data on the 'Time' column
combined_data = pd.merge(temperature_data, power_data, on='Time', how='inner')

# Save the combined data to a new CSV file
combined_data.to_csv('combined_data.csv', index=False)

print("Combined data saved in 'combined_data.csv'")
