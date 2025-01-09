# Import necessary libraries
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Step 1: Create a connection to SQLite database
con = sqlite3.connect("my_data1.db")
cur = con.cursor()

# Step 2: Load the SpaceX dataset into a pandas DataFrame
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")

# Display dataset structure for verification
print("Dataset Columns:", df.columns)
print("First few rows of the dataset:")
print(df.head())

# Step 3: Load the DataFrame into the SQLite database
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False, method="multi")

# Step 4: Check if the table SPACEXTABLE exists and drop it if necessary
cur.execute("DROP TABLE IF EXISTS SPACEXTABLE;")

# Step 5: Create SPACEXTABLE with non-null dates
cur.execute("CREATE TABLE SPACEXTABLE AS SELECT * FROM SPACEXTBL WHERE Date IS NOT NULL;")


# Step 6: Analyze the data with SQL queries
# Display the first 10 rows
print("First 10 rows of SPACEXTABLE:")
for row in cur.execute("SELECT * FROM SPACEXTABLE LIMIT 10;"):
    print(row)



# Step 6: Analyze the data with SQL queries
# Display the first 10 rows
print("First 10 rows of SPACEXTABLE:")
for row in cur.execute("SELECT * FROM SPACEXTABLE LIMIT 10;"):
    print(row)

# Display the names of the unique launch sites in the space mission:
print("\nDisplay the names of the unique launch sites in the space mission:")
for row in cur.execute("SELECT DISTINCT Launch_Site FROM SPACEXTBL;"):
    print(row)

# Display 5 records where launch sites begin with the string 'KSC'
print("\nDisplay 5 records where launch sites begin with the string 'KSC'")
for row in cur.execute("SELECT * FROM SPACEXTBL WHERE Launch_Site LIKE 'KSC%' LIMIT 5;"):
    print(row)

# Display the total payload mass carried by boosters launched by NASA (CRS)
print("\nDisplay the total payload mass carried by boosters launched by NASA (CRS)")
for row in cur.execute("SELECT SUM(PAYLOAD_MASS__KG_) AS Total_Payload_Mass FROM SPACEXTBL WHERE Payload LIKE '%CRS%';"):
    print(row)

# Display average payload mass carried by booster version F9 v1.1
print("\nDisplay average payload mass carried by booster version F9 v1.1")
for row in cur.execute("SELECT AVG(PAYLOAD_MASS__KG_) AS Average_Payload_Mass FROM SPACEXTBL WHERE Booster_Version = 'F9 v1.1';"):
    print(row)

# List the date where the succesful landing outcome in drone ship was acheived
print("\nList the date where the succesful landing outcome in drone ship was acheived")
for row in cur.execute("SELECT DISTINCT Date FROM SPACEXTBL WHERE Landing_Outcome = 'Success (drone ship)';"):
    print(row)

# List the names of the boosters which have success in ground pad and have payload mass greater than 4000 but less than 6000
print("\nList the names of the boosters which have success in ground pad and have payload mass greater than 4000 but less than 6000")
for row in cur.execute("SELECT DISTINCT Booster_Version FROM SPACEXTBL WHERE Landing_Outcome = 'Success (ground pad)'  AND PAYLOAD_MASS__KG_ > 4000   AND PAYLOAD_MASS__KG_ < 6000;"):
    print(row) 

# List the total number of successful and failure mission outcomes
print("\nList the total number of successful and failure mission outcomes")
for row in cur.execute("SELECT Mission_Outcome, COUNT(*) AS Total_Count FROM SPACEXTBL GROUP BY Mission_Outcome;"):
    print(row) 

# List the names of the booster_versions which have carried the maximum payload mass. Use a subquery
print("\nList the names of the booster_versions which have carried the maximum payload mass. Use a subquery")
for row in cur.execute("SELECT DISTINCT Booster_Version FROM SPACEXTBL WHERE PAYLOAD_MASS__KG_ = (SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTBL);"):
    print(row) 

# List the records which will display the month names, succesful landing_outcomes in ground pad ,booster versions, launch_site for the months in year 2017
print("\nList the names of the booster_versions which have carried the maximum payload mass. Use a subquery")
for row in cur.execute("""SELECT 
    CASE substr(Date, 6, 2)
        WHEN '01' THEN 'January'
        WHEN '02' THEN 'February'
        WHEN '03' THEN 'March'
        WHEN '04' THEN 'April'
        WHEN '05' THEN 'May'
        WHEN '06' THEN 'June'
        WHEN '07' THEN 'July'
        WHEN '08' THEN 'August'
        WHEN '09' THEN 'September'
        WHEN '10' THEN 'October'
        WHEN '11' THEN 'November'
        WHEN '12' THEN 'December'
    END AS MonthName,
    landing_outcome,
    booster_version,
    launch_site
FROM 
    SPACEXTBL
WHERE 
    substr(Date, 0, 5) = '2017'
    AND landing_outcome LIKE '%ground pad%'
ORDER BY 
    MonthName;
"""):
    print(row) 

# Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order
print("\nRank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order")
for row in cur.execute("""SELECT 
    landing_outcome,
    COUNT(*) AS outcome_count,
    RANK() OVER (ORDER BY COUNT(*) DESC) AS rank
FROM 
    SPACEXTBL
WHERE 
    Date BETWEEN '2010-06-04' AND '2017-03-20'
GROUP BY 
    "Landing_Outcome"
ORDER BY 
    outcome_count DESC;"""):
    print(row) 

# Count total number of launches
print("\nTotal number of launches:")
cur.execute("SELECT COUNT(*) AS Total_Launches FROM SPACEXTABLE;")
print(cur.fetchone())

# Group launches by launch site
print("\nLaunches grouped by Launch Site:")
for row in cur.execute("SELECT Launch_Site, COUNT(*) AS Launch_Count FROM SPACEXTABLE GROUP BY Launch_Site;"):
    print(row)

# Success rate by launch site
print("\nSuccess rate by Launch Site:")
for row in cur.execute("""
SELECT Launch_Site, 
       COUNT(CASE WHEN Mission_Outcome
 LIKE '%Success%' THEN 1 END) * 100.0 / COUNT(*) AS Success_Rate
FROM SPACEXTABLE
GROUP BY Launch_Site;
"""):
    print(row)

# Frequency of payload types
print("\nFrequency of payload types:")
for row in cur.execute("""
SELECT Payload
, COUNT(*) AS Frequency 
FROM SPACEXTABLE 
GROUP BY Payload
ORDER BY Frequency DESC;
"""):
    print(row)

# Launches by year
print("\nLaunches by year:")
for row in cur.execute("""
SELECT SUBSTR(Date, 1, 4) AS Year, COUNT(*) AS Launches 
FROM SPACEXTABLE 
GROUP BY Year 
ORDER BY Year;
"""):
    print(row)

# Step 7: Visualizations
# Visualization: Launches by Site
query = "SELECT Launch_Site, COUNT(*) AS Launch_Count FROM SPACEXTABLE GROUP BY Launch_Site;"
launch_data = pd.read_sql(query, con)

launch_data.plot(kind='bar', x='Launch_Site', y='Launch_Count', legend=False)
plt.title('Launches by Site')
plt.ylabel('Launch Count')
plt.xlabel('Launch Site')
plt.show()

# Visualization: Success Rate by Launch Site
query = """
SELECT Launch_Site, 
       COUNT(CASE WHEN Mission_Outcome
 LIKE '%Success%' THEN 1 END) * 100.0 / COUNT(*) AS Success_Rate
FROM SPACEXTABLE
GROUP BY Launch_Site;
"""
success_data = pd.read_sql(query, con)

success_data.plot(kind='bar', x='Launch_Site', y='Success_Rate', legend=False, color='green')
plt.title('Success Rate by Launch Site')
plt.ylabel('Success Rate (%)')
plt.xlabel('Launch Site')
plt.show()

# Step 8: Close the database connection
con.close()

# Count unique values in each categorical column
n1 = df['Orbit'].nunique()
n2 = df['Launch_Site'].nunique()
n3 = df['LandingPad'].nunique()
n4 = df['Serial'].nunique()

# Calculate total columns
total_columns = len(df.columns) - 4 + (n1 + n2 + n3 + n4)
print("Total Columns:", total_columns)
