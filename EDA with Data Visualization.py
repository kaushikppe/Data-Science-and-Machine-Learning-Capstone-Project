# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
df = pd.read_csv(URL)

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head(5))

# Summary statistics of the dataset
print("\nDataset summary:")
print(df.describe())

# Plot 1: FlightNumber vs. PayloadMass with Outcome (Class) Overlay
print("\nVisualizing the relationship between FlightNumber, PayloadMass, and Class:")
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Payload Mass (kg)", fontsize=20)
plt.title("Flight Number vs. Payload Mass and Class", fontsize=22)
plt.show()

# Plot 2: Launch Success Rate by Site
print("\nVisualizing the success rate of launches for each site:")
launch_success_rate = df.groupby("LaunchSite")["Class"].mean().reset_index()
sns.barplot(x="LaunchSite", y="Class", data=launch_success_rate, palette="Blues_d")
plt.xlabel("Launch Site", fontsize=16)
plt.ylabel("Success Rate", fontsize=16)
plt.title("Launch Success Rate by Site", fontsize=18)
plt.show()

# Plot 3: Orbit vs. Success Rate
print("\nVisualizing success rates across different orbits:")
orbit_success_rate = df.groupby("Orbit")["Class"].mean().reset_index()
sns.barplot(x="Orbit", y="Class", data=orbit_success_rate, palette="Greens_d")
plt.xlabel("Orbit", fontsize=16)
plt.ylabel("Success Rate", fontsize=16)
plt.title("Success Rate by Orbit", fontsize=18)
plt.xticks(rotation=45)
plt.show()

# Plot 4: Payload Mass vs. Success for Specific Orbits
print("\nAnalyzing Payload Mass vs. Success for specific orbits:")
selected_orbits = ["GTO", "LEO", "SSO"]
df_filtered = df[df["Orbit"].isin(selected_orbits)]
sns.scatterplot(x="PayloadMass", y="Class", hue="Orbit", data=df_filtered)
plt.xlabel("Payload Mass (kg)", fontsize=16)
plt.ylabel("Success (Class)", fontsize=16)
plt.title("Payload Mass vs. Success for Selected Orbits", fontsize=18)
plt.legend(title="Orbit")
plt.show()

# Plot 5: Yearly Launch Success Trends
print("\nVisualizing launch success trends over the years:")
df["Year"] = pd.DatetimeIndex(df["Date"]).year
yearly_success_rate = df.groupby("Year")["Class"].mean().reset_index()
sns.lineplot(x="Year", y="Class", data=yearly_success_rate, marker="o")
plt.xlabel("Year", fontsize=16)
plt.ylabel("Success Rate", fontsize=16)
plt.title("Yearly Launch Success Trends", fontsize=18)
plt.show()

# Plot 6: Correlation Heatmap
print("\nVisualizing correlations between numeric features:")
numeric_columns = ["FlightNumber", "PayloadMass", "Block", "ReusedCount", "Class"]
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=18)
plt.show()

#TASK 1: Visualize the relationship between Flight Number and Launch Site
import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatter plot using Seaborn's catplot
sns.catplot(
    x="FlightNumber",       # X-axis: Flight Number
    y="LaunchSite",         # Y-axis: Launch Site
    hue="Class",            # Hue: Class (success or failure)
    data=df,                # Data source
    aspect=2,               # Stretch the plot horizontally
    kind="strip"            # Scatter plot style
)

# Customize the plot
plt.xlabel("Flight Number", fontsize=15)   # Label for X-axis
plt.ylabel("Launch Site", fontsize=15)     # Label for Y-axis
plt.title("Flight Number vs. Launch Site", fontsize=18)  # Plot title
plt.show()                                 # Display the plot

#TASK 2: Visualize the relationship between Payload Mass and Launch Site
import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatter plot using Seaborn's catplot
sns.catplot(
    x="PayloadMass",       # X-axis: Payload Mass (kg)
    y="LaunchSite",        # Y-axis: Launch Site
    hue="Class",           # Hue: Class (success or failure)
    data=df,               # Data source
    aspect=2,              # Stretch the plot horizontally
    kind="strip"           # Scatter plot style
)

# Customize the plot
plt.xlabel("Payload Mass (kg)", fontsize=15)  # Label for X-axis
plt.ylabel("Launch Site", fontsize=15)        # Label for Y-axis
plt.title("Payload Mass vs. Launch Site", fontsize=18)  # Plot title
plt.show()                                    # Display the plot

#TASK 3: Visualize the relationship between success rate of each orbit type
# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Group the data by Orbit and calculate the mean success rate for each orbit type
orbit_success_rate = df.groupby('Orbit')['Class'].mean().reset_index()

# Sort the values by success rate for better visualization
orbit_success_rate = orbit_success_rate.sort_values(by='Class', ascending=False)

# Create a bar plot to visualize the success rate for each Orbit
plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Orbit', data=orbit_success_rate, palette='Blues_d')

# Customize the plot
plt.xlabel("Success Rate", fontsize=15)  # Label for X-axis
plt.ylabel("Orbit Type", fontsize=15)    # Label for Y-axis
plt.title("Success Rate for Each Orbit Type", fontsize=18)  # Plot title
plt.show()

#TASK 4: Visualize the relationship between FlightNumber and Orbit type
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatter plot
sns.scatterplot(
    data=df, 
    x='FlightNumber', 
    y='Orbit', 
    hue='Class', 
    palette={0: "red", 1: "green"},  # Map: 0 -> red (Failure), 1 -> green (Success)
    s=100                            # Marker size
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

# Create a scatter plot for Payload Mass vs. Orbit Type
sns.scatterplot(
    data=df, 
    x='PayloadMass', 
    y='Orbit', 
    hue='Class', 
    palette={0: "red", 1: "green"},  # 0: Failure (red), 1: Success (green)
    s=100                             # Marker size
)

# Add labels and title
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Payload Mass vs Orbit Type (Class)", fontsize=16)
plt.legend(title='Class', loc='upper right', labels=['Failure', 'Success'])
plt.grid(True)
plt.show()

#TASK 6: Visualize the launch success yearly trend
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Select relevant features
features = df[['FlightNumber', 'PayloadMass', 'LaunchSite', 'Orbit', 'Year']]

# One-hot encode categorical variables
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(features[['LaunchSite', 'Orbit']]).toarray()

# Get feature names from the encoder
encoded_feature_names = encoder.get_feature_names_out(['LaunchSite', 'Orbit'])

# Create a DataFrame with encoded features
encoded_features_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Combine numerical features with encoded features
final_features = pd.concat([features[['FlightNumber', 'PayloadMass', 'Year']], encoded_features_df], axis=1)

# Display the final engineered features
final_features.head()

#TASK 8: Cast all numeric columns to float64
# Select categorical columns for one-hot encoding
features_one_hot = pd.get_dummies(df[['LaunchSite', 'Orbit']])

# Cast the columns to 'float64'
features_one_hot = features_one_hot.astype('float64')

# Confirm the data type conversion
print(features_one_hot.dtypes)

# Optionally, export to CSV for further use
features_one_hot.to_csv('dataset_part_3.csv', index=False)
