import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data from the heart database
connect = sqlite3.connect('heart.db')
cur = connect.cursor()
cur.execute('SELECT * FROM heart_data')
data = cur.fetchall()
columns = [desc[0] for desc in cur.description]
df = pd.DataFrame(data, columns=columns)

# Defining the target variable
target_var = 'target'

# Defining the categorical variables
numeric_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Create a figure and axis object
fig, axs = plt.subplots(nrows=len(numeric_vars), ncols=1, figsize=(8, 20))

# Loop through each numeric variable
for i, var in enumerate(numeric_vars):
    # Plot the distribution of classes for the current variable
    axs[i].hist(df[df[target_var] == 0][var], alpha=0.5, label='Class 0')
    axs[i].hist(df[df[target_var] == 1][var], alpha=0.5, label='Class 1')
    axs[i].set_title(f'Distribution of {var} by {target_var}')
    axs[i].legend()

# Show the plot
plt.show()