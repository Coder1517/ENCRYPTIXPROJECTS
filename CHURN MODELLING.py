#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the dataset
# Replace 'your_data.csv' with the actual path to your CSV file
df = pd.read_csv('Churn_Modelling.csv')

# Step 2: Display the initial state of the dataset (before filling missing values)
print("Missing values before cleaning:")
print(df.isnull().sum())  # Shows the count of missing values in each column

# Step 3: Handle missing values
# For categorical columns (non-numeric), fill with the mode (most frequent value)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    mode_value = df[col].mode()[0]  # Get the mode (most frequent value)
    df[col].fillna(mode_value, inplace=True)  # Fill missing with mode

# For numerical columns (integers and floats), fill with the mean
numerical_columns = df.select_dtypes(include=['number']).columns
for col in numerical_columns:
    mean_value = df[col].mean()  # Get the mean value
    df[col].fillna(mean_value, inplace=True)  # Fill missing with mean

# Step 4: Verify that missing values are filled
print("\nMissing values after cleaning:")
print(df.isnull().sum())  # Check if all missing values have been filled

# Step 5: Display the dataset to verify the changes
print("\nFirst few rows of the cleaned dataset:")
print(df.head())  # Print the first few rows to check if the missing values are filled

# Step 6:  Check the column names to make sure they are correct
print("\nColumn names in the dataset:")
print(df.columns)

# Step 7: Convert categorical variables to numeric using One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)  # This creates binary columns for categorical variables

# Step 8: Prepare the data for machine learning
# Use 'Exited' as the target variable
X = df.drop(columns=['Exited'])  # Features
y = df['Exited']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 10: Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Feature importance
print("\nFeature Importances:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Step 11: Save the trained model (optional)
import joblib
joblib.dump(model, 'random_forest_model.pkl')  # Save the model to a file

print("\nModel training complete and saved as 'random_forest_model.pkl'")


# In[ ]:




