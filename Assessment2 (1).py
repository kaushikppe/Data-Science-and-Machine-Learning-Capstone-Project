import requests
from bs4 import BeautifulSoup
import unicodedata
import pandas as pd

# Helper functions
def date_time(table_cells):
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    out = ''.join([booster_version for i, booster_version in enumerate(table_cells.strings) if i % 2 == 0][0:-1])
    return out

def landing_status(table_cells):
    out = [i for i in table_cells.strings][0]
    return out

def get_mass(table_cells):
    mass = unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass = mass[0:mass.find("kg") + 2]
    else:
        new_mass = 0
    return new_mass

def extract_column_from_header(row):
    if row.br:
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name

# URL for static Wikipedia page snapshot
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

# TASK 1: Request the Falcon 9 Launch Wiki page
response = requests.get(static_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Verify the page title
print("Page title:", soup.title.string)

# TASK 2: Extract column names from the HTML table header
html_tables = soup.find_all('table', class_='wikitable')
first_launch_table = html_tables[2]

column_names = []
for th in first_launch_table.find_all('th'):
    name = extract_column_from_header(th)
    if name is not None and len(name) > 0:
        column_names.append(name)

print("Extracted Column Names:", column_names)

# TASK 3: Create and populate the data dictionary
launch_dict = dict.fromkeys(column_names)
del launch_dict['Date and time ( )']

launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
launch_dict['Version Booster'] = []
launch_dict['Booster landing'] = []
launch_dict['Date'] = []
launch_dict['Time'] = []

extracted_row = 0
for table_number, table in enumerate(soup.find_all('table', "wikitable plainrowheaders collapsible")):
    for rows in table.find_all("tr"):
        if rows.th:
            if rows.th.string:
                flight_number = rows.th.string.strip()
                flag = flight_number.isdigit()
        else:
            flag = False
        
        if flag:
            extracted_row += 1
            row = rows.find_all('td')
            
            # Extract and append data to the dictionary
            launch_dict['Flight No.'].append(flight_number)
            datatimelist = date_time(row[0])
            launch_dict['Date'].append(datatimelist[0].strip(','))
            launch_dict['Time'].append(datatimelist[1])
            launch_dict['Version Booster'].append(booster_version(row[1]) or row[1].a.string)
            launch_dict['Launch site'].append(row[2].a.string)
            launch_dict['Payload'].append(row[3].a.string)
            launch_dict['Payload mass'].append(get_mass(row[4]))
            launch_dict['Orbit'].append(row[5].a.string)
            launch_dict['Customer'].append(row[6].a.string)
            launch_dict['Launch outcome'].append(list(row[7].strings)[0])
            launch_dict['Booster landing'].append(landing_status(row[8]))

# Create a DataFrame and save to CSV
df = pd.DataFrame({key: pd.Series(value) for key, value in launch_dict.items()})
df.to_csv('spacex_web_scraped.csv', index=False)
print("Data saved to spacex_web_scraped.csv")
