# Import required libraries
import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import pandas as pd

# Helper functions
def date_time(table_cells):
    """Extract date and time from a table cell."""
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """Extract booster version from a table cell."""
    out = ''.join([booster_version for i, booster_version in enumerate(table_cells.strings) if i % 2 == 0][0:-1])
    return out

def landing_status(table_cells):
    """Extract landing status from a table cell."""
    out = [i for i in table_cells.strings][0]
    return out

def get_mass(table_cells):
    """Extract payload mass from a table cell."""
    mass = unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass = mass[0:mass.find("kg") + 2]
    else:
        new_mass = None
    return new_mass

def extract_column_from_header(row):
    """Extract column name from a table header row."""
    if row.br:
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
    colunm_name = ' '.join(row.contents)
    if not colunm_name.strip().isdigit():
        return colunm_name.strip()

# URL for the Falcon 9 and Falcon Heavy launches
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

# Request the HTML page
response = requests.get(static_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Verify page title
print("Page Title:", soup.title.string)

# Extract all tables
html_tables = soup.find_all('table', "wikitable plainrowheaders collapsible")

# Extract column names from the third table
first_launch_table = html_tables[2]
column_names = []
for th in first_launch_table.find_all('th'):
    name = extract_column_from_header(th)
    if name is not None and len(name) > 0:
        column_names.append(name)

print("Extracted Column Names:")
print(column_names)

# Create a dictionary for launch data
launch_dict = dict.fromkeys(column_names)
launch_dict = {key: [] for key in column_names}
launch_dict['Version Booster'] = []
launch_dict['Booster landing'] = []
launch_dict['Date'] = []
launch_dict['Time'] = []

# Parse table rows and extract data
extracted_row = 0
for table_number, table in enumerate(html_tables):
    for rows in table.find_all("tr"):
        if rows.th and rows.th.string:
            flight_number = rows.th.string.strip()
            flag = flight_number.isdigit()
        else:
            flag = False

        row = rows.find_all('td')
        if flag:
            extracted_row += 1
            # Flight Number
            launch_dict['Flight No.'].append(flight_number)

            # Date and Time
            datatimelist = date_time(row[0])
            launch_dict['Date'].append(datatimelist[0].strip(',') if len(datatimelist) > 0 else None)
            launch_dict['Time'].append(datatimelist[1] if len(datatimelist) > 1 else None)

            # Booster Version
            bv = booster_version(row[1])
            if not bv:
                bv = row[1].a.string if row[1].a else None
            launch_dict['Version Booster'].append(bv)

            # Launch Site
            launch_dict['Launch site'].append(row[2].a.string if row[2].a else None)

            # Payload
            launch_dict['Payload'].append(row[3].a.string if row[3].a else None)

            # Payload Mass
            launch_dict['Payload mass'].append(get_mass(row[4]))

            # Orbit
            launch_dict['Orbit'].append(row[5].a.string if row[5].a else None)

            # Customer
            customer = row[6].a.string if row[6].a else None
            launch_dict['Customer'].append(customer)

            # Launch Outcome
            launch_outcome = list(row[7].strings)[0] if row[7].strings else None
            launch_dict['Launch outcome'].append(launch_outcome)

            # Booster Landing
            booster_landing = landing_status(row[8]) if len(row) > 8 else None
            launch_dict['Booster landing'].append(booster_landing)

# Ensure all columns have the same length by padding missing values
max_length = max(len(v) for v in launch_dict.values())
for key in launch_dict:
    while len(launch_dict[key]) < max_length:
        launch_dict[key].append(None)

# Create DataFrame
df = pd.DataFrame(launch_dict)

# Save to CSV
df.to_csv('spacex_web_scraped.csv', index=False)

print("Web scraping completed. Data saved to 'spacex_web_scraped.csv'.")
