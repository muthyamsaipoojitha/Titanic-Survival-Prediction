# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 2: Load the Titanic dataset from the specified path
file_path = r'C:\Users\saira\Downloads\Titanic-Survival-Prediction\data\tested.csv'
data = pd.read_csv(file_path)

# Step 3: Preprocess the data
# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Drop columns that are not useful for prediction
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 4: Encode categorical variables
X = data.drop('Survived', axis=1)
y = data['Survived']

# Define the preprocessing steps
numerical_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Pclass', 'Embarked']

# Preprocessing for numerical data (normalize)
numerical_transformer = StandardScaler()

# Preprocessing for categorical data (one-hot encode)
categorical_transformer = OneHotEncoder(drop='first')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 5: Build the model pipeline
model = RandomForestClassifier(n_estimators=100, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Step 6: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
pipeline.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = pipeline.predict(X_test)

# Print out the evaluation metrics
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Optionally, use cross-validation to get a better estimate
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print('Cross-validation Accuracy:', cv_scores.mean())
