import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, classification_report

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

# Getting the column names from the categorical and numerical variables
column_names = []
for cat in cat_vars:
    column_names.extend([f"{cat}_{i}" for i in range(preprocessing_pipeline.named_transformers_['cat']['onehot'].categories_[cat_vars.index(cat)].shape[0])])
for num in num_vars:
    column_names.append(num)

# Converting the preprocessed data to a pandas DataFrame
df_preprocessed = pd.DataFrame(df_preprocessed, columns=column_names)

# Adding the 'target' column to the DataFrame
df_preprocessed['target'] = df[target_var]

# Spliting the preprocessed data into features and target
X = df_preprocessed.drop('target', axis=1)
y = df_preprocessed['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the scaler within the preprocessing pipeline
scaler = preprocessing_pipeline.named_transformers_['num']['scaler']


# Defining the models that will be used
models = [
    LogisticRegression(),
    RandomForestClassifier(n_estimators=100),
    SVC(kernel='rbf', C=1)
]

# Fitting each model to the training data
for model in models:
    model.fit(X_train, y_train)

# Performing predictions on the test data
y_pred_lr = models[0].predict(X_test)
y_pred_rf = models[1].predict(X_test)
y_pred_svm = models[2].predict(X_test)

# Evaluating the performance of each model
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Support Vector Machine:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Saving the best-performing model to disk
# The Random Forest model performed the best
joblib.dump(models[1], 'heart_disease_model.joblib')
joblib.dump(scaler, 'scaler.joblib')