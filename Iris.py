import pandas as pd
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
import numpy as np

# Unzip and load the dataset
with zipfile.ZipFile('/mnt/data/iris.zip', 'r') as zip_ref:
    zip_ref.extractall('/mnt/data/')

file_path = '/mnt/data/iris.csv'  # Adjust the file path based on the extracted file
iris_df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Info:")
print(iris_df.info())
print("\nDataset Description:")
print(iris_df.describe())
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())

# Exploratory Data Analysis (EDA)
# Visualize relationships between features and species
sns.pairplot(iris_df, hue='species')
plt.show()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Data Preprocessing
# Encode species labels to numeric values
le = LabelEncoder()
iris_df['species'] = le.fit_transform(iris_df['species'])

# Split the data into training and testing sets
X = iris_df.drop(columns='species')
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} - Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

# Select the best model based on performance
best_model = models['Random Forest']

# Deploy the model using Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from POST request and predict species
    data = request.get_json(force=True)
    features = np.array([data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]).reshape(1, -1)
    prediction = best_model.predict(features)
    species = le.inverse_transform(prediction)[0]
    return jsonify({'species': species})

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True)
