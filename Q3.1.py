# Importing the necessary libraries
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Loading the data and establishing a connection
connect = sqlite3.connect('heart.db')
cur = connect.cursor()
cur.execute('SELECT * FROM heart_data')
data = cur.fetchall()
columns = [desc[0] for desc in cur.description]
df = pd.DataFrame(data, columns=columns)

# Defining the target variable
target_var = 'target'

# Defining the categorical and numerical variables
cat_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
num_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Creating a preprocessing pipeline for categorical variables
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Creating a preprocessing pipeline for numerical variables
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Creating a preprocessing pipeline for the entire dataset
preprocessing_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, cat_vars),
    ('num', num_pipeline, num_vars)
])

# Appling the preprocessing pipeline to the dataset
df_preprocessed = preprocessing_pipeline.fit_transform(df)

# Spliting the preprocessed data into features and target
X = df_preprocessed[:, :-1]
y = df_preprocessed[:, -1]

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The data has been preprocessed and is able to fit a machine learning model.