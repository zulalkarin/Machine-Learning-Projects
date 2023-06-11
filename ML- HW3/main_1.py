# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns


# Define column names (according to the dataset documentation)
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\Mchine HW3\heart+disease\processed.cleveland.data', names=columns, na_values='?') # adjust the file path accordingly

# Handle missing values (if any)
df = df.dropna()

# In the original dataset, 'target' > 1 means that the patient has heart disease. We will convert this to a binary classification task: '0' for no disease and '1' for disease.
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Split data into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)])

# Create a pipeline for each model
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier())])

svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC())])

logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression())])

# Train and evaluate each model
pipelines = [rf_pipeline, svc_pipeline, logreg_pipeline]
pipeline_names = ['Random Forest', 'Support Vector Machine', 'Logistic Regression']

for i, pipe in enumerate(pipelines):
    scores = cross_val_score(pipe, X_train, y_train, cv=10)
    print(f'{pipeline_names[i]}: Average cross-validation score: {scores.mean()}')

# Perform hyperparameter tuning on the best model
param_grid = {'classifier__n_estimators': [50, 100, 150],
              'classifier__max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=10)
grid_search.fit(X_train, y_train)
print()
print(f'Best parameters: {grid_search.best_params_}')
print()
print(f'Best score: {grid_search.best_score_}')
print()

# Test the model
y_pred = grid_search.predict(X_test)
print(f'Test set accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Test set precision: {precision_score(y_test, y_pred)}')
print(f'Test set recall: {recall_score(y_test, y_pred)}')
print(f'Test set F1-score: {f1_score(y_test, y_pred)}')


# # Create a scatter plot of 'age' and 'chol' colored by 'target'
# sns.scatterplot(data=df, x='age', y='chol', hue='target')
# plt.title('Age vs Cholesterol Level-2')
# plt.show()
#
# # Create a scatter plot of 'age' and 'trestbps'
# sns.scatterplot(data=df, x='age', y='trestbps', hue='target')
# plt.title('Age vs Resting Blood Pressure')
# plt.show()
#
# # Create a count plot of 'cp'
# sns.countplot(data=df, x='cp')
# plt.title('Chest Pain Type')
# plt.show()
#
# # Create a count plot of 'thal'
# sns.countplot(data=df, x='thal')
# plt.title('Thalassemia Type')
# plt.show()


# # Calculate average cross-validation scores
# cross_val_scores = [0.7887, 0.8133, 0.8257]
#
# # Create a bar plot for model comparison
# plt.figure(figsize=(8, 6))
# plt.bar(pipeline_names, cross_val_scores)
# plt.xlabel('Models')
# plt.ylabel('Average Cross-Validation Score')
# plt.title('Comparison of Model Performance')
# plt.show()

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Create a confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()