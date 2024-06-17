# wine-ML
Wine Quality Prediction with Random Forest Classifier
This project aims to predict the quality of wine based on various physicochemical properties using a Random Forest Classifier.

Dataset
The dataset (data.csv) contains information on different physicochemical properties of wine and their respective quality ratings.

Workflow
Data Loading and Exploration:

The dataset is loaded into a Pandas DataFrame (wine_data), and basic information such as dimensions, missing values, and statistical summaries are displayed.
Visualizations are created to understand the distribution of wine quality and its relationship with various features (e.g., volatile acidity, citric acid).
Data Cleaning and Preprocessing:

The dataset is checked for missing values and summary statistics are computed.
Correlation analysis is performed using a heatmap to understand the relationships between different features.
The target variable (quality) is binarized into two classes: good quality (1) for ratings >= 7 and bad quality (0) for ratings < 7.
Feature Selection:

Features (X) are selected by dropping the 'quality' column from the dataset.
The target (Y) is created by applying a lambda function to binarize the wine quality ratings.
Model Training and Evaluation:

The data is split into training and test sets using train_test_split.
A Random Forest Classifier is trained on the training data.
The model's performance is evaluated using accuracy scores on both training and test datasets.
Prediction:

A predictive system is built to classify new wine data based on the trained model.
Example predictions are made for new input data, and the model predicts whether the wine is of good or bad quality.
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn (sklearn)
Usage
Ensure Python and necessary libraries (pandas, numpy, matplotlib, seaborn, scikit-learn) are installed.
Clone this repository and navigate to the project directory.
Place your dataset (data.csv) in the project directory.
Run the script wine_quality_prediction.py to load the dataset, preprocess data, train the Random Forest Classifier, and predict wine quality based on new data instances.
Example
Upon running the script, the Random Forest Classifier utilizes physicochemical properties of wine (such as acidity, sugar content, and alcohol) to predict wine quality. The accuracy of predictions on both training and test datasets provides an assessment of the model's reliability in classifying wine quality.

