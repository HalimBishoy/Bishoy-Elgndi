# Bishoy-Elgndi
#Project Module 2  CIND 820
# Final Project_Module_Two: Red Wine Quality:
import numpy as np
import pandas as pd

import seaborn as sns
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Reading File: 
Red_Wine = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
Red_Wine

#Checking the dimesions of the dataset
print(Red_Wine.shape)

#Checking the data types of the attributes in the dataset:
Red_Wine.info()
print("\n")

# Checking the missing values of the attributes of the dataset:
print("Number of missing values in each column\n")
print(Red_Wine.isnull().sum())

# All 12 columns are of numeric data types. Out of 12 variables, 11 are predictor variables and last one 'quality' is an response/target variable.
#Let's look at the summary of the dataset,
Red_Wine.describe()

#The features of the target/response variable.
#The scores obtained are between 3 to 8.
Red_Wine['quality'].unique()

#Checking how many unique value does the target variabble 'quality' has?
Red_Wine.quality.value_counts().sort_index()

#Plotting the frequency distribution of the wine quality using barplot
sns.countplot(x='quality', data=Red_Wine)

#The frequency distribution of wine quality using Piechart
import plotly.express as px
Red_Wine_new = Red_Wine['quality'].value_counts().rename_axis('Winequality').reset_index(name='counts')
Red_Wine_new
fig = px.pie(Red_Wine_new, values='counts', names='Winequality')
fig.show()

#The correlation between the target/response variable (Quality) and the predictor variables:
correlation = Red_Wine.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")

# Sorting correlation values for all attributes with quality 
correlation['quality'].sort_values(ascending=False)

#Checking the correleation between the variables using Seaborn's pairplot. 
sns.pairplot(Red_Wine)

# Histogram
Red_Wine.hist(bins=10,figsize=(8,10))
plt.show()

#Plotting the boxplots to check the Outliers for each column against the target variable
sns.boxplot('quality', 'fixed acidity', data = Red_Wine)
sns.boxplot('quality', 'volatile acidity', data = Red_Wine)
sns.boxplot('quality', 'citric acid', data = Red_Wine)
sns.boxplot('quality', 'residual sugar', data = Red_Wine)
sns.boxplot('quality', 'chlorides', data = Red_Wine)
sns.boxplot('quality', 'free sulfur dioxide', data = Red_Wine)
sns.boxplot('quality', 'total sulfur dioxide', data = Red_Wine)
sns.boxplot('quality', 'density', data = Red_Wine)
sns.boxplot('quality', 'pH', data = Red_Wine)
sns.boxplot('quality', 'sulphates', data = Red_Wine)
sns.boxplot('quality', 'alcohol', data = Red_Wine)

