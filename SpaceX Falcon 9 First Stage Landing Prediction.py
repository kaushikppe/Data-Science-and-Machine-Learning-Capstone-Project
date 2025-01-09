# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests  # Use requests to fetch the dataset

# Fetch the dataset from the URL
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
response = requests.get(URL)
dataset_part_2_csv = io.BytesIO(response.content)

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(dataset_part_2_csv)

# Display the first 5 rows
print(df.head(5))

# Create a categorical plot using Seaborn
sns.catplot(
    y="PayloadMass",              # Payload mass (kg) on the y-axis
    x="FlightNumber",             # Flight number on the x-axis
    hue="Class",                  # Differentiate by class (e.g., success/failure)
    data=df,                      # Data source
    aspect=5                      # Stretch the plot horizontally
)

# Customize axis labels
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Payload Mass (kg)", fontsize=20)

# Display the plot
plt.show()

#TASK 1: Visualize the relationship between Flight Number and Launch Site
import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatter plot with Seaborn catplot
sns.catplot(
    x="FlightNumber",       # X-axis: Flight Number
    y="LaunchSite",         # Y-axis: Launch Site
    hue="Class",            # Hue: Class (success or failure)
    data=df,                # Data source
    aspect=2,               # Stretch the plot horizontally
    kind="strip"            # Scatter plot style
)

# Customize the plot
plt.xlabel("Flight Number", fontsize=15)
plt.ylabel("Launch Site", fontsize=15)
plt.title("Flight Number vs. Launch Site", fontsize=18)
plt.show()

#TASK 2: Visualize the relationship between Payload Mass and Launch Site
# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatter plot with Seaborn catplot
sns.catplot(
    x="PayloadMass",       # X-axis: Payload Mass (kg)
    y="LaunchSite",        # Y-axis: Launch Site
    hue="Class",           # Hue: Class (success or failure)
    data=df,               # Data source
    aspect=2,              # Stretch the plot horizontally
    kind="strip"           # Scatter plot style
)

# Customize the plot
plt.xlabel("Payload Mass (kg)", fontsize=15)
plt.ylabel("Launch Site", fontsize=15)
plt.title("Payload Mass vs. Launch Site", fontsize=18)
plt.show()

#TASK 3: Visualize the relationship between success rate of each orbit type
# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Group the data by Orbit and calculate the mean success rate for each orbit type
orbit_success_rate = df.groupby('Orbit')['Class'].mean().reset_index()

# Sort the values by success rate for better visualization
orbit_success_rate = orbit_success_rate.sort_values(by='Class', ascending=False)

# Create a bar plot to visualize the success rate for each Orbit
sns.barplot(x='Class', y='Orbit', data=orbit_success_rate, palette='Blues_d')

# Customize the plot
plt.xlabel("Success Rate", fontsize=15)
plt.ylabel("Orbit Type", fontsize=15)
plt.title("Success Rate for Each Orbit Type", fontsize=18)
plt.show()

# Group by Orbit and calculate the success rate (mean of Class)
orbit_success_rate = df.groupby('Orbit')['Class'].mean()

# Plot the bar chart
plt.figure(figsize=(10, 6))
orbit_success_rate.plot(kind='bar', color='skyblue')

# Customize the plot
plt.title("Success Rate by Orbit Type", fontsize=18)
plt.xlabel("Orbit Type", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.show()

#TASK 4: Visualize the relationship between FlightNumber and Orbit type
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=df, 
    x='FlightNumber', 
    y='Orbit', 
    hue='Class', 
    palette={0: "red", 1: "green"},  # Explicit mapping: 0 -> red, 1 -> green
    s=100
)

# Add labels and title
plt.xlabel("Flight Number", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Flight Number vs Orbit Type (Class)", fontsize=16)
plt.legend(title='Class', loc='upper right', labels=['Failure', 'Success'])
plt.grid(True)
plt.show()

#TASK 5: Visualize the relationship between Payload Mass and Orbit type
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot for Payload Mass vs. Orbit Type
sns.scatterplot(
    data=df, 
    x='PayloadMass', 
    y='Orbit', 
    hue='Class', 
    palette={0: "red", 1: "green"},  # 0: Failure (red), 1: Success (green)
    s=100
)

# Add labels and title
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Payload Mass vs Orbit Type (Class)", fontsize=16)
plt.legend(title='Class', loc='upper right', labels=['Failure', 'Success'])
plt.grid(True)
plt.show()

#TASK 6: Visualize the launch success yearly trend
# Function to extract years from the date
def Extract_year():
    year = []
    for i in df["Date"]:
        year.append(i.split("-")[0])  # Extract the year part from the date string
    return year

# Apply the function and create a 'Year' column
df['Year'] = Extract_year()

# Group by 'Year' and calculate the average success rate
yearly_success_rate = df.groupby('Year')['Class'].mean().reset_index()

# Rename columns for clarity
yearly_success_rate.rename(columns={'Class': 'SuccessRate'}, inplace=True)

# Convert 'Year' to integer for proper plotting
yearly_success_rate['Year'] = yearly_success_rate['Year'].astype(int)

# Plot the line chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_success_rate, x='Year', y='SuccessRate', marker='o', color='blue')

# Add labels and title
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Success Rate", fontsize=14)
plt.title("Yearly Launch Success Trend", fontsize=16)
plt.xticks(yearly_success_rate['Year'], rotation=45)
plt.grid(True)
plt.show()

#TASK 7: Create dummy variables to categorical columns
# Select the relevant features
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 
               'ReusedCount', 'Serial']]

# Apply one-hot encoding to categorical columns
features_one_hot = pd.get_dummies(
    features, 
    columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial']
)

# Display the first few rows of the new dataframe
features_one_hot.head()
print(features_one_hot.columns)
print(features_one_hot.shape)


#TASK 8: Cast all numeric columns to float64
# Cast all numeric columns to float64
features_one_hot = features_one_hot.astype('float64')

# Export the dataframe to a CSV file for use in the next lab
features_one_hot.to_csv('dataset_part_3.csv', index=False)

# Confirm the data type conversion
print(features_one_hot.dtypes)
