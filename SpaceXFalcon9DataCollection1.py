!pip install requests pandas numpy
import requests
import pandas as pd
import numpy as np
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def getBoosterVersion(data):
    for x in data['rocket']:
        if x:
            response = requests.get(f"https://api.spacexdata.com/v4/rockets/{x}").json()
            BoosterVersion.append(response['name'])


def getLaunchSite(data):
    for x in data['launchpad']:
        if x:
            response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{x}").json()
            Longitude.append(response['longitude'])
            Latitude.append(response['latitude'])
            LaunchSite.append(response['name'])


def getPayloadData(data):
    for load in data['payloads']:
        if load:
            response = requests.get(f"https://api.spacexdata.com/v4/payloads/{load}").json()
            PayloadMass.append(response['mass_kg'])
            Orbit.append(response['orbit'])


def getCoreData(data):
    for core in data['cores']:
        if core['core']:
            response = requests.get(f"https://api.spacexdata.com/v4/cores/{core['core']}").json()
            Block.append(response['block'])
            ReusedCount.append(response['reuse_count'])
            Serial.append(response['serial'])
        else:
            Block.append(None)
            ReusedCount.append(None)
            Serial.append(None)
        Outcome.append(str(core['landing_success']) + ' ' + str(core['landing_type']))
        Flights.append(core['flight'])
        GridFins.append(core['gridfins'])
        Reused.append(core['reused'])
        Legs.append(core['legs'])
        LandingPad.append(core['landpad'])


spacex_url = "https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)

# Check response status
print(response.status_code)

# Convert JSON response to Pandas DataFrame
data = pd.json_normalize(response.json())
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]


data = data[data['cores'].map(len) == 1]
data = data[data['payloads'].map(len) == 1]


data['cores'] = data['cores'].map(lambda x: x[0])
data['payloads'] = data['payloads'].map(lambda x: x[0])
data['date'] = pd.to_datetime(data['date_utc']).dt.date
data = data[data['date'] <= datetime.date(2020, 11, 13)]


# Initialize variables
BoosterVersion, PayloadMass, Orbit, LaunchSite = [], [], [], []
Outcome, Flights, GridFins, Reused, Legs, LandingPad = [], [], [], [], [], []
Block, ReusedCount, Serial, Longitude, Latitude = [], [], [], [], []

# Populate variables
getBoosterVersion(data)
getLaunchSite(data)
getPayloadData(data)
getCoreData(data)



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
    'Latitude': Latitude,
}
final_df = pd.DataFrame(launch_dict)


data_falcon9 = final_df[final_df['BoosterVersion'] != 'Falcon 1']
data_falcon9.loc[:, 'FlightNumber'] = list(range(1, data_falcon9.shape[0] + 1))


data_falcon9.to_csv('dataset_part_1.csv', index=False)

