import folium
import pandas as pd
import requests
from io import BytesIO

# Import folium plugins
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon

# Download and read the `spacex_launch_geo.csv`
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
response = requests.get(URL)
spacex_csv_file = BytesIO(response.content)
spacex_df = pd.read_csv(spacex_csv_file)

# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]

# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]

#TASK 1: Mark all launch sites on a map
# Initialize the map centered at NASA JSC
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

# Add a circle and marker for NASA Johnson Space Center
nasa_circle = folium.Circle(
    nasa_coordinate, 
    radius=1000, 
    color='#d35400', 
    fill=True
).add_child(folium.Popup('NASA Johnson Space Center'))

nasa_marker = folium.Marker(
    nasa_coordinate,
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>NASA JSC</b></div>'
    )
)

site_map.add_child(nasa_circle)
site_map.add_child(nasa_marker)

# Add circles and markers for each launch site
for _, row in launch_sites_df.iterrows():
    # Extract the coordinates and site name
    site_name = row['Launch Site']
    lat = row['Lat']
    long = row['Long']
    
    # Add a circle for the launch site
    folium.Circle(
        location=(lat, long),
        radius=1000,  # 1 km radius
        color='#007849',  # Green border
        fill=True,
        fill_color='#39FF14',  # Light green fill
        fill_opacity=0.5
    ).add_child(
        folium.Popup(f"Launch Site: {site_name}")
    ).add_to(site_map)
    
    # Add a marker for the launch site
    folium.Marker(
        location=(lat, long),
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12px; color:#d35400;"><b>{site_name}</b></div>'
        )
    ).add_to(site_map)

# Save the map as an HTML file or display it directly in a Jupyter Notebook
site_map.save("launch_sites_map.html")


# Task 2: Mark the success/failed launches for each site on the map
# Initialize the map
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

# Add launch site circles and markers
for _, row in launch_sites_df.iterrows():
    site_name = row['Launch Site']
    lat = row['Lat']
    long = row['Long']
    
    # Add a circle for each launch site
    folium.Circle(
        location=(lat, long),
        radius=1000,  # 1 km radius
        color='#007849',  # Green border
        fill=True,
        fill_color='#39FF14',  # Light green fill
        fill_opacity=0.5
    ).add_child(
        folium.Popup(f"Launch Site: {site_name}")
    ).add_to(site_map)
    
    # Add a marker for the launch site
    folium.Marker(
        location=(lat, long),
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12px; color:#d35400;"><b>{site_name}</b></div>'
        )
    ).add_to(site_map)

# Add markers for each launch outcome
for _, row in spacex_df.iterrows():
    lat = row['Lat']
    long = row['Long']
    launch_class = row['class']
    
    # Determine marker color based on class
    marker_color = 'green' if launch_class == 1 else 'red'
    launch_status = 'Success' if launch_class == 1 else 'Failure'
    
    # Add a marker for the launch
    folium.Marker(
        location=(lat, long),
        icon=folium.Icon(color=marker_color, icon='info-sign'),
        popup=f"Launch Site: {row['Launch Site']}<br>Status: {launch_status}"
    ).add_to(site_map)

# Save the map as an HTML file
site_map.save("launch_outcomes_map.html")

# Display the map
site_map

# TASK 3: Calculate the distances between a launch site to its proximities

import folium
import pandas as pd
from folium.plugins import MarkerCluster
from folium.features import DivIcon

# Load the SpaceX dataset (assuming spacex_df is already loaded)
# spacex_df contains columns: Launch Site, Lat, Long, class

# Create a MarkerCluster object
marker_cluster = MarkerCluster()

# Add a column to spacex_df to determine marker colors based on launch outcome
def assign_marker_color(launch_class):
    if launch_class == 1:
        return 'green'
    else:
        return 'red'

spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)

# Initialize the map centered at an arbitrary coordinate
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

# Add markers to the marker_cluster
for index, record in spacex_df.iterrows():
    # Extract the latitude and longitude of the launch
    launch_location = [record['Lat'], record['Long']]
    # Extract the marker color
    marker_color = record['marker_color']
    # Create a folium.Marker and add it to the marker cluster
    marker = folium.Marker(
        location=launch_location,
        icon=folium.Icon(color=marker_color, icon="info-sign"),
        popup=folium.Popup(f"Launch Site: {record['Launch Site']}<br>Class: {record['class']}", parse_html=True)
    )
    marker_cluster.add_child(marker)

# Add the marker_cluster to the map
site_map.add_child(marker_cluster)

# Save the map to an HTML file or display it
site_map.save("launch_sites_map_with_outcomes.html")
site_map

from math import sin, cos, sqrt, atan2, radians
import folium
from folium.features import DivIcon

# Function to calculate the distance between two coordinates using Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    # Approximate radius of Earth in km
    R = 6373.0

    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Example launch site and closest coastline coordinates
launch_site_lat = 28.56342
launch_site_lon = -80.57674
coastline_lat = 28.56367
coastline_lon = -80.57163

# Calculate the distance
distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)

# Create a folium map centered around the launch site
site_map = folium.Map(location=[launch_site_lat, launch_site_lon], zoom_start=14)

# Add a marker for the launch site
launch_marker = folium.Marker(
    [launch_site_lat, launch_site_lon],
    popup="Launch Site",
    icon=folium.Icon(color="blue", icon="info-sign"),
)
site_map.add_child(launch_marker)

# Add a marker for the closest coastline
distance_marker = folium.Marker(
    [coastline_lat, coastline_lon],
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>{:10.2f} KM</b></div>'.format(distance_coastline),
    ),
)
site_map.add_child(distance_marker)

# Draw a line between the launch site and the coastline point
line = folium.PolyLine(
    locations=[[launch_site_lat, launch_site_lon], [coastline_lat, coastline_lon]], weight=2, color="blue"
)
site_map.add_child(line)

# Save the map to an HTML file or display it
site_map.save("launch_site_to_coastline_map.html")
site_map


from math import sin, cos, sqrt, atan2, radians
import folium
from folium.features import DivIcon

# Function to calculate the distance between two coordinates using Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6373.0  # Approximate radius of Earth in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Example launch site coordinates
launch_site_lat = 28.56342
launch_site_lon = -80.57674

# Coordinates for the closest city, railway, and highway
city_lat, city_lon = 28.4019, -80.6057    # Example: Titusville, FL
railway_lat, railway_lon = 28.5721, -80.5853  # Example: Closest railway point
highway_lat, highway_lon = 28.5623, -80.5774  # Example: Closest highway point

# Calculate distances
distance_to_city = calculate_distance(launch_site_lat, launch_site_lon, city_lat, city_lon)
distance_to_railway = calculate_distance(launch_site_lat, launch_site_lon, railway_lat, railway_lon)
distance_to_highway = calculate_distance(launch_site_lat, launch_site_lon, highway_lat, highway_lon)

# Create a folium map centered at the launch site
site_map = folium.Map(location=[launch_site_lat, launch_site_lon], zoom_start=12)

# Add a marker for the launch site
launch_marker = folium.Marker(
    [launch_site_lat, launch_site_lon],
    popup="Launch Site",
    icon=folium.Icon(color="blue", icon="info-sign"),
)
site_map.add_child(launch_marker)

# Add markers for city, railway, and highway with distances
locations = [
    ("Closest City", city_lat, city_lon, distance_to_city, "green"),
    ("Closest Railway", railway_lat, railway_lon, distance_to_railway, "red"),
    ("Closest Highway", highway_lat, highway_lon, distance_to_highway, "purple"),
]

for label, lat, lon, distance, color in locations:
    marker = folium.Marker(
        [lat, lon],
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12; color:{color};"><b>{label}<br>{distance:.2f} KM</b></div>',
        ),
    )
    site_map.add_child(marker)

    # Draw a line from the launch site to the point
    line = folium.PolyLine(
        locations=[[launch_site_lat, launch_site_lon], [lat, lon]], weight=2, color=color
    )
    site_map.add_child(line)

# Save the map to an HTML file or display it
site_map.save("launch_site_to_points_map.html")
site_map


from folium.plugins import MousePosition

# Add MousePosition plugin to the map
mouse_position = MousePosition(
    position='topright',
    separator=', ',
    prefix="Coordinates:",
    lat_lon_digits=6
)
site_map.add_child(mouse_position)

import folium
from folium.features import DivIcon
from folium.plugins import MousePosition
from math import sin, cos, sqrt, atan2, radians

# Function to calculate distance using Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6373.0  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Launch site coordinates
launch_site_name = "CCAFS LC-40"  # Example
launch_site_lat = 28.56342
launch_site_lon = -80.57674

# Create the map centered at the launch site
site_map = folium.Map(location=[launch_site_lat, launch_site_lon], zoom_start=12)

# Add MousePosition plugin for coordinate detection
mouse_position = MousePosition(position='topright', separator=', ', prefix="Coordinates:", lat_lon_digits=6)
site_map.add_child(mouse_position)

# Add a marker for the launch site
launch_marker = folium.Marker(
    [launch_site_lat, launch_site_lon],
    popup=launch_site_name,
    icon=folium.Icon(color="blue", icon="info-sign"),
)
site_map.add_child(launch_marker)

# Proximity coordinates (replace these with coordinates from MousePosition)
proximities = [
    ("Closest City", 28.4019, -80.6057, "green"),  # Example: Titusville
    ("Closest Railway", 28.5721, -80.5853, "red"),  # Example: Closest railway
    ("Closest Highway", 28.5623, -80.5774, "purple"),  # Example: Closest highway
    ("Closest Coastline", 28.56367, -80.57163, "orange"),  # Example coastline
]

# Add markers and draw lines
for name, lat, lon, color in proximities:
    distance = calculate_distance(launch_site_lat, launch_site_lon, lat, lon)

    # Add a marker for the proximity point
    proximity_marker = folium.Marker(
        [lat, lon],
        icon=DivIcon(
            icon_size=(20, 20),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12; color:{color};"><b>{name}<br>{distance:.2f} KM</b></div>',
        ),
    )
    site_map.add_child(proximity_marker)

    # Draw a line from the launch site to the proximity point
    line = folium.PolyLine(locations=[[launch_site_lat, launch_site_lon], [lat, lon]], weight=2, color=color)
    site_map.add_child(line)

# Save and display the map
site_map.save("launch_proximities_map.html")
site_map
