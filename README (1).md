![](UTA-DataScience-Logo.png)

# Liver Cirrhosis Predictions

- **One Sentence Summary** The repository holds an attempt using machine learning models to predict liver cirrhosis outcomes.

## Overview

The Liver Cirrhosis Kaggle Challenge is a machine learning competition focused on predicting the presence of liver cirrhosis in patients based on various medical features and data. The goal is to develop machine learning models that can accurately classify patients as either having liver cirrhosis or not.

- **Your approach**
  THe best model that I had was Random Forrest and XGBoost.

## Summary of Workdone

### Data

- Data:
  - Is from Mayo Clininc and contains 17 features for patients.
  - The data goes over the states of patients.
  - With 0 being that the patient is dead, 1 being the patient is censored and 2 being that the patient was censored due to the patient having a liver transplant.
  - Size: The data set had a size of 1.39 MB.

#### Preprocessing / Clean up

- I visualized the data first to look for skewness and checked for any missing or duplicate values. I created boxplots to check for outliers and removed them. In addition to this, I encoded the categorical variables to make it easir to visualze my data.

#### Data Visualization

Data visualization before any preprocessing
![Alt Text](https://github.com/anishg2001/DATA3402.Spring.2024/blob/main/inintial%20data%20boxplots.png)
![Alt Text](https://github.com/anishg2001/DATA3402.Spring.2024/blob/main/boxplot%20of%20cirrhosis.png)
Data Visualization after Processing
![Alt Text](https://github.com/anishg2001/DATA3402.Spring.2024/blob/main/data%20viz%20after.png)
Feature Engineering
![Alt Text](https://github.com/anishg2001/DATA3402.Spring.2024/blob/main/red.png)

### Problem Formulation

- Define:

  - Models
    - I tried several different models: Random Forest, KNN, SVM, and XGBoost.
    - I used these models because I wanted to see which ones would run better and have better metrics.

### Performance Comparison

- After completing my, the project I did not see much change in the visualization, I had completed before and after.
- I believe there is further work to be done in my data transformation and preprocesssing.

### Conclusions

- FOr my conclusion, I believe there are other more ML algorithims that needs to be run. Liver cirrhosis is a complex medical condition with significant implications for patient health and well-being. Due to its complexity, I believe more models need to be ran in predicting to find the model that works best.

### Future Work

- For future work, I would need to add some more hypertuning and data transformation to help increase the accuracy of my model.

## How to reproduce results

- To reproduce my results, first you would have to download the dataset into your IDE and download the libraires in my notebook.
- After downloading the proper libraires, data unterstand will need to be completed.
- After completing data understanding, a preliminary data visualziation needs to be completed.
- After completing some basic visualization, you can review the visualizations to look for outliers and duplicate data.
- After doing the necessary data transformation, you can run the ML models on your data.

### Overview of files in repository

- There are two files in my repository
- The first one being the notebook with my code and output and the second is the read.md file that is the conclusion of my results.

### Software Setup

- I used google collaboratory as my IDE and downloaded the libraries into the collaboratory file.
- Libraries used:
  import seaborn as sns
  from sklearn.svm import SVC
  from tabulate import tabulate
  import matplotlib.pylab as plt

  from xgboost import XGBClassifier
  from sklearn.preprocessing import LabelEncoder
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score,classification_report
  from sklearn.model_selection import train_test_split, GridSearchCV

### Data

- The data can be downloaded at the provided link: https://www.kaggle.com/competitions/playground-series-s3e26/code
-

### Training

- To train the module I used an 80/20 split. I used 80 percent of the data for training and 20 for the test. I did not use the sample submission data.

#### Performance Evaluation

- After running multiple models, the two best models were Random Forest and XGclassifier both of them had an accuracy of 83%.

## Citations

- Walter Reade, Ashley Chow. (2023). Multi-Class Prediction of Cirrhosis Outcomes. Kaggle. https://kaggle.com/competitions/playground-series-s3e26
