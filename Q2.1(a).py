import sqlite3
import pandas as pd

# Establishing a connection to the database
conn = sqlite3.connect('heart.db')

# Loading the data from the CSV file
df = pd.read_csv('heart.csv')

# Checking for missing values
print(df.isnull().sum())

# Droping rows with missing values
df.dropna(inplace=True)

# Checking for duplicates
print('Total duplicates:',df.duplicated().sum())

# Droping duplicates
df.drop_duplicates(inplace=True)

# Checking for duplicates
print('Total duplicates, after dropping the duplicates:',df.duplicated().sum())

# Commit the changes
conn.commit()

# Close the connection
conn.close()