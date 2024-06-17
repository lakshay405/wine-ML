import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading the dataset from CSV file into a Pandas DataFrame
wine_data = pd.read_csv('data.csv')

# Checking the dimensions of the dataset
print("Shape of the dataset:", wine_data.shape)

# Displaying the first few rows of the dataset
wine_data.head()

# Checking for missing values in each column
print("Missing values in the dataset:")
print(wine_data.isnull().sum())

# Displaying statistical summary of the dataset
print("Statistical summary of the dataset:")
print(wine_data.describe())

# Plotting the count of each quality level
sns.catplot(x='quality', data=wine_data, kind='count')

# Plotting volatile acidity vs Quality
plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='volatile acidity', data=wine_data)

# Plotting citric acid vs Quality
plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='citric acid', data=wine_data)

# Calculating correlation matrix
correlation = wine_data.corr()

# Constructing a heatmap to understand the correlation between columns
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'fontsize': 12}, cmap='Blues')

# Separating features (X) and target (Y) variables
X = wine_data.drop('quality', axis=1)
print("Features (X):")
print(X.head())

Y = wine_data['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
print("Target (Y):")
print(Y.head())

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Displaying test set labels
print("Test set labels (Y_test):")
print(Y_test.head())

# Displaying train set labels
print("Train set labels (Y_train):")
print(Y_train.head())

# Model training - Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Model evaluation
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data:', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data:', test_data_accuracy)

# Predictive model
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)

# Converting input data to a numpy array
input_data_np = np.asarray(input_data)

# Reshaping the input data for prediction
input_data_reshaped = input_data_np.reshape(1, -1)

# Making prediction
prediction = model.predict(input_data_reshaped)
print(prediction)

# Displaying prediction result
if prediction[0] == 1:
    print('Predicted: Good Quality Wine')
else:
    print('Predicted: Bad Quality Wine')
