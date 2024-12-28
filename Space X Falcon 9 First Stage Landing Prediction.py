import pandas as pd
import numpy as np
import io
from urllib.request import urlopen

# Define the URL
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv'

# Open the URL and read the data
response = urlopen(URL)
dataset_part_1_csv = io.BytesIO(response.read())

# Read CSV data into a DataFrame
df = pd.read_csv(dataset_part_1_csv)

# Display the first 10 rows
print(df.head(10))

# Check for missing values (percentage)
print(df.isnull().sum() / df.shape[0] * 100)

# Display data types of each column
print(df.dtypes)

# Apply value_counts on the 'LaunchSite' column to see the number of launches for each site
launch_site_counts = df['LaunchSite'].value_counts()
print("Launch Site Counts:")
print(launch_site_counts)

# Apply value_counts on the 'Orbit' column to see the number of occurrences of each orbit type
orbit_counts = df['Orbit'].value_counts()
print("Orbit Counts:")
print(orbit_counts)

# Apply value_counts on the 'Outcome' column to see the number of occurrences of each outcome type
landing_outcomes = df['Outcome'].value_counts()
print("Landing Outcomes:")
print(landing_outcomes)

for i, outcome in enumerate(landing_outcomes.keys()):
    print(i, outcome)
# Assuming you have already calculated landing_outcomes using value_counts on the Outcome column
landing_outcomes = df['Outcome'].value_counts()

# Create the bad_outcomes set by selecting specific indices
bad_outcomes = set([landing_outcomes.keys()[i] for i in [1, 3, 5, 6, 7]])

# Output the bad_outcomes
print(bad_outcomes)

# Create the list 'landing_class'
landing_class = [0 if outcome in bad_outcomes else 1 for outcome in df['Outcome']]

# Assign the 'landing_class' list to the new column 'Class'
df['Class'] = landing_class

# Show the first 8 rows of the 'Class' column
print(df[['Class']].head(8))

# Show the first 5 rows of the DataFrame to check overall data
print(df.head(5))

# Calculate and print the mean of the 'Class' column
mean_class = df["Class"].mean()
print(f"Mean of 'Class' column: {mean_class}")

# Save the updated DataFrame to a CSV file
df.to_csv(r"dataset_part_2.csv", index=False)
