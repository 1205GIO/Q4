import sqlite3
import pandas as pd
import csv

# Opening the CSV file with the correct delimiter
with open('heart.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')

    # Creating a new database connection
    conn = sqlite3.connect('heart.db')

    print("Connection established")

    # Creating a cursor object
    cur = conn.cursor()

    # Creating a new table called 'heart_data'
    cur.execute('''
        CREATE TABLE IF NOT EXISTS heart_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age numerical,
            sex nominal,
            cp nominal,
            trestbps numerical,
            chol numerical,
            fbs nominal,
            restecg nominal,
            thalach numerical,
            exang nominal,
            oldpeak numerical,
            slope nominal,
            ca nominal,
            thal nominal,
            target nominal
        )
    ''')

    print("Table has been created")

    # Skip the header row
    next(reader)

    # Inserting records into the table
    cur.executemany('''
            INSERT INTO heart_data (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target)
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', reader)

    # Commit the changes
    conn.commit()

    print("Records have been inserted")

    # Close the connection
    conn.close()