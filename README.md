# COVID-19 Hospital Length of Stay Prediction

## Overview

This project predicts hospital length-of-stay categories for COVID-19 patients using hospital admission, patient, ward, illness severity, and utilization-related features. The goal is to help healthcare teams identify patients who may require longer stays and support earlier planning for beds, staffing, and care coordination.

## Objective

Predict the `stay_days` category using machine learning classification models.

## Dataset

The dataset contains 318K+ hospital admission records with 18 columns, including:

* Hospital and region details
* Department and ward information
* Admission type
* Illness severity
* Age group
* Patient visitors
* Admission deposit
* Length of stay category

## Tools Used

* Python
* PySpark
* Spark MLlib
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## Project Workflow

1. Loaded and inspected hospital admission data.
2. Cleaned column names and converted data types.
3. Encoded categorical variables using StringIndexer.
4. Created feature vectors using VectorAssembler.
5. Applied PCA and feature scaling.
6. Split the data into training and testing sets.
7. Trained Decision Tree and Random Forest classifiers.
8. Evaluated models using classification metrics.
9. Analyzed length-of-stay patterns by illness severity, ward type, department, and age group.

## Models Used

* Decision Tree Classifier
* Random Forest Classifier

## Evaluation Metrics

* Accuracy
* F1 Score
* Weighted Precision
* Weighted Recall
* Confusion Matrix
* Feature Importance

## Key Insights

* Most patients stayed between 11 and 30 days.
* Illness severity, admission type, ward type, department, and age group were important factors for predicting length of stay.
* Random Forest provided slightly stronger performance than Decision Tree.
* Length-of-stay prediction can support hospital capacity planning, resource allocation, and early care coordination.

## Business Value

This project shows how healthcare data science can support operational planning by predicting patient stay duration earlier in the admission process. These insights can help hospitals manage bed capacity, staffing needs, and high-risk patient monitoring more effectively.

## Future Improvements

* Add advanced models such as Gradient Boosted Trees, XGBoost, or LightGBM.
* Tune model hyperparameters for better performance.
* Add SHAP or feature importance analysis for model explainability.
* Build a dashboard to monitor length-of-stay risk groups.
* Package the model for batch prediction or API-based deployment.
