# Different Models Used
# 1. BMI-Calculator
File Location -> bmi_calculator.py<br>

Body Mass Index (BMI) is a simple and widely used method for assessing an individual's body weight relative to their height. It is calculated by dividing a person's weight in kilograms by the square of their height in meters (kg/m²). Here are some reasons why BMI is useful:
The BMI categories used in this application are:
- Underweight: BMI < 18.5
- Normal weight: 18.5 <= BMI < 24.9
- Overweight: 25 <= BMI < 29.9
- Obese: BMI >= 30

# 2. Diabetes Predictor 
File Location -> colab_files_to_train_models -> Multiple disease prediction system - diabetes.ipynb<br>
Dataset Location -> dataset -> diabetes.csv

The algorithm being developed for this research uses machine learning to forecast an individual's risk of developing diabetes based on a number of different health indicators. Using Streamlit as the UI framework, the application lets users enter pertinent health information and get estimates of their risk of developing diabetes<br>
Features<br>
- Input health parameters including glucose levels, blood pressure, skin thickness, insulin levels, BMI, age, etc.
- Predict the likelihood of diabetes based on the input parameters.
- User-friendly interface built with Streamlit.
- Visualize the results and health data easily.

This Model is build over Support Vector Classifier, A Machine Learning Model and Provides a accuracy of 77% over the test data

# 3. Heart Disease Predictor
File Location -> colab_files_to_train_models -> Multiple disease prediction - heart.ipynb<br>
Dataset Location -> dataset -> heart.csv

This project is a machine learning-based system that uses a variety of health metrics to estimate an individual's risk of heart disease. Using Streamlit as the UI framework, the application lets users enter pertinent health information and get risk estimates for heart disease.<br>
Features<br>
- Input health parameters including age, sex, blood pressure, cholesterol levels, and more.
- Predict the likelihood of heart disease based on the input parameters.
- User-friendly interface built with Streamlit.
- Visualize the results and health data easily.
- Proper Exploratory Data Analysis has been done to find about important attributes.<br>

Following machine learning models has been tested<br>
- KNN + tune hyperparameters

- SVM + tune hyperparameters

- Decision Trees + tune hyperparameters

- Random Forest + tune hyperparameters

Finally support vector classifier has been selected with a maximum accuracy of 93%<br>

# 4. Parkinsons Disease Predictor
File Location -> colab_files_to_train_models -> Multiple disease prediction - Parkinsons.ipynb<br>
Dataset Location -> dataset -> parkinsons.csv

With the use of several health indicators, this project's machine learning-based system will be able to forecast a person's risk of developing Parkinson's disease. Using Streamlit as the UI framework, the program lets users enter pertinent health information and get estimates of their chance of developing Parkinson's disease.<br>
Features<br>
- input health parameters including voice measurements and other relevant medical data.
- Proper Exploratory Data Analysis has been done to find the similarity between various types of attributes also the the impact of each feature on the probability of the positive class.
- Predict the likelihood of Parkinson's disease based on the input parameters.
- User-friendly interface built with Streamlit.
- Visualize the results and health data easily.

Following Machine Learning Models has been used to determine the outcome for Disease
- k-Nearest Neighbors (k-NN)
- GridSearchCV
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest Model

Final Model used to for classification is Random Forest Classifier with accuracy of 89%

# 5. Breast Cancer Classifier (Malignant/Benign)
File Location -> cancer_prediction.py
Dataset Location -> data.csv

Having various Health indicators as attributes like a cell's mean radius, Texture Mean, perimeter Mean, Area mean, etc this Machine Learning Model predicts whether the breast cancer is of malignant or benign type.<br>
Features<br>
- input health parameters including cell's Texture and other relevant data from Data.csv file
- Exploratory Data Analysis has been done.
- Standarization of data also has been done to improve the model's accuracy.
- User-friendly interface built with Streamlit.
- Visualization of result with best accuracy of whether the person is suffering from malignant or benign tumour.

Machine Learning Model used -> Logistic Regression<br>
Accuracy of model -> 97%
