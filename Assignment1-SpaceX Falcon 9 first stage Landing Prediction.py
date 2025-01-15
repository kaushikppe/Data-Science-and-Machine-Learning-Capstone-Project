import requests
import pandas as pd
import numpy as np
import datetime

# Setting display options for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Define helper functions
def getBoosterVersion(data):
    """Extracts BoosterVersion details using the rocket column."""
    for x in data['rocket']:
        if x:
            response = requests.get(f"https://api.spacexdata.com/v4/rockets/{x}").json()
            BoosterVersion.append(response['name'])

def getLaunchSite(data):
    """Extracts launch site details using the launchpad column."""
    for x in data['launchpad']:
        if x:
            response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{x}").json()
            Longitude.append(response.get('longitude', None))
            Latitude.append(response.get('latitude', None))
            LaunchSite.append(response.get('name', None))

def getPayloadData(data):
    """Extracts payload data (mass and orbit) using the payloads column."""
    for load in data['payloads']:
        if load and isinstance(load, list):  # Ensure load is a non-empty list
            payload_id = load[0]  # Extract the first payload ID
            response = requests.get(f"https://api.spacexdata.com/v4/payloads/{payload_id}").json()
            PayloadMass.append(response.get('mass_kg', None))
            Orbit.append(response.get('orbit', None))
        else:
            PayloadMass.append(None)
            Orbit.append(None)

def getCoreData(data):
    """Extracts core-related details using the cores column."""
    for core in data['cores']:
        if core['core'] is not None:
            response = requests.get(f"https://api.spacexdata.com/v4/cores/{core['core']}").json()
            Block.append(response.get('block', None))
            ReusedCount.append(response.get('reuse_count', None))
            Serial.append(response.get('serial', None))
        else:
            Block.append(None)
            ReusedCount.append(None)
            Serial.append(None)
        Outcome.append(str(core['landing_success']) + ' ' + str(core['landing_type']))
        Flights.append(core.get('flight', None))
        GridFins.append(core.get('gridfins', None))
        Reused.append(core.get('reused', None))
        Legs.append(core.get('legs', None))
        LandingPad.append(core.get('landpad', None))

# Global variables for storing data
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []

# Step 1: Fetch and parse SpaceX launch data
static_json_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'

response = requests.get(static_json_url)
if response.status_code == 200:
    print("API request successful")
else:
    print("API request failed")

# Decode JSON content
data = pd.json_normalize(response.json())
print("Data fetched successfully!")

# Display first 5 rows
print(data.head())

# Step 2: Subset and clean the data
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

# Filter out rows with multiple cores and payloads
data = data[data['cores'].map(len) == 1]
data = data[data['payloads'].map(len) == 1]

# Flatten lists in cores and payloads columns
data['cores'] = data['cores'].map(lambda x: x[0])
data['payloads'] = data['payloads'].map(lambda x: x[0])

# Convert date_utc to datetime and extract date
data['date'] = pd.to_datetime(data['date_utc']).dt.date

# Restrict the dataset to dates before November 13, 2020
data = data[data['date'] <= datetime.date(2020, 11, 13)]

# Step 3: Extract additional data using API calls
getBoosterVersion(data)
getLaunchSite(data)
getPayloadData(data)
getCoreData(data)

# Step 4: Create a new dictionary for the processed data
launch_dict = {
    'FlightNumber': list(data['flight_number']),
    'Date': list(data['date']),
    'BoosterVersion': BoosterVersion,
    'PayloadMass': PayloadMass,
    'Orbit': Orbit,
    'LaunchSite': LaunchSite,
    'Outcome': Outcome,
    'Flights': Flights,
    'GridFins': GridFins,
    'Reused': Reused,
    'Legs': Legs,
    'LandingPad': LandingPad,
    'Block': Block,
    'ReusedCount': ReusedCount,
    'Serial': Serial,
    'Longitude': Longitude,
    'Latitude': Latitude
}

# Step 5: Create a dataframe from the dictionary
final_data = pd.DataFrame(launch_dict)

# Step 6: Display the summary of the dataframe
print("Summary of the dataset:")
print(final_data.info())
print(final_data.head())


# Task 2: Filter the dataframe to only include Falcon 9 launches
# Filter the data to only include rows where BoosterVersion is not 'Falcon 1'
data_falcon9 = final_data[final_data['BoosterVersion'] != 'Falcon 1']

# Reset the FlightNumber column to be a sequential range starting from 1
data_falcon9.loc[:, 'FlightNumber'] = list(range(1, data_falcon9.shape[0] + 1))

# Display the filtered dataframe
print("Filtered DataFrame (Falcon 9 launches only):")
print(data_falcon9.head())

# Task 3: Dealing with Missing Values
# Check for missing values
print("Missing values before handling:")
print(data_falcon9.isnull().sum())

# Calculate the mean value of the PayloadMass column
payload_mass_mean = data_falcon9['PayloadMass'].mean()

# Replace np.nan values in PayloadMass with the calculated mean
data_falcon9['PayloadMass'] = data_falcon9['PayloadMass'].replace(np.nan, payload_mass_mean)

# Verify that missing values in PayloadMass are handled
print("\nMissing values after handling:")
print(data_falcon9.isnull().sum())

# Export the cleaned dataset to a CSV file
output_file = 'dataset_part_1.csv'
data_falcon9.to_csv(output_file, index=False)
print(f"\nCleaned dataset exported to {output_file}")
