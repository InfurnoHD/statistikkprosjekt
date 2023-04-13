import pandas as pd

# read the original CSV file
data = pd.read_csv("NO2_2021.csv", sep=";")
data1 = pd.read_csv("NO2_2022.csv", sep=";")

# create a list to store the expanded rows
expanded_rows = []

# iterate over each row in the DataFrame
for index, row in data.iterrows():

    # extract the date from the "Dato" column
    date = row["Dato"]

    # iterate over the "Time1" to "Time24" columns
    for hour in range(1, 25):
        # extract the consumption value for the current hour
        consumption = row[f"Time{hour}"]

        # construct the new "referenceTime" value with the hour included
        reference_time = f"{date}T{hour - 1:02d}:00:00.000Z"

        # create a new dictionary with the expanded row data
        new_row = {"referenceTime": reference_time, "consumption": consumption}

        # add the new row to the list
        expanded_rows.append(new_row)

for index, row in data1.iterrows():

    # extract the date from the "Dato" column
    date = row["Dato"]

    # iterate over the "Time1" to "Time24" columns
    for hour in range(1, 25):
        # extract the consumption value for the current hour
        consumption = row[f"Time{hour}"]

        # construct the new "referenceTime" value with the hour included
        reference_time = f"{date}T{hour - 1:02d}:00:00.000Z"

        # create a new dictionary with the expanded row data
        new_row = {"referenceTime": reference_time, "consumption": consumption}

        # add the new row to the list
        expanded_rows.append(new_row)

# create a new DataFrame with the expanded rows
expanded_data = pd.DataFrame(expanded_rows)

# write the new DataFrame to a CSV file
expanded_data.to_csv("power.csv", index=False)
