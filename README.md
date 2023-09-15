
Goal is to predict Smokers and Drinkers using body signal data. 

•	Manual Observation in identifying variables and  getting familiar with the dataset 

Dataset contains medical records of individual between age 20 to 85. The records consist of details of physical examination such as height, weight, sight, hearing, BP and other health signals through blood test of cholesterol, liver function. Kidney etc.   

Using health statistic, we are trying to classify individual as smokers (may be they quit smoking regular smokers, chain smokers, etc.) and also check if they are consuming alcohol . 

We can use physical examination data to identify outliers values and remove that data. 

We can correlate the HDL, LDL, total cholesterol, haemoglobin data values to smokers, and the kidney and liver function tests (SGoT, urine protein etc.) to drinkers.

And based on new test records of an individual we may try to predict what type of smoker he or she would be.  


•	Exploratory Data Analysis (EDA) 

EDA purpose is to understand the dataset, cleanse it and analyse the relationship between variable. 

Import libraries such as numpy, pandas, matplotlib, seaborn etc.  for the analysis. 

.shape returns the number of rows by the number of columns for my dataset. My output was (991346, 24), meaning the dataset has 991346rows and 24 columns. 

.head() returns the first 5 rows of my dataset. This is useful if you want to see some example values for each variable. 

https://github.com/SaarthChahal/ML-DL/blob/main/first%205%20rows%20of%20dataset.png

.columns returns the name of all of your columns in the dataset. 

https://github.com/SaarthChahal/ML-DL/blob/main/dataset%20columns.png

After this I worked on getting better understanding of the different values for each variable. 

.nunique(axis=0) returns the number of unique values for each variable.

.describe() summarizes the count, mean, standard deviation, min, and max for numeric variables

https://github.com/SaarthChahal/ML-DL/blob/main/describe%20dataset.png
 
data.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))) output 



**Cleaning your DataSet by removing outliers, nulls.**


Used .dropna(axis=0) to remove any rows with null values. There was no null value so cleaned data still returned the same (991346, 24) for data_cleaned.shape 
Removed outliers by using varaible.between( lower limit, upper limit ) and variable < limit 
Data for analysis does not use string datatype as an argument , I could have encoded the gender variable labelled as ‘sex’ to number for male or female or I could have dropped column itself. I choose to drop the column by using .drop(‘variable’) 
Also I encoded /converted DRK_YN variable string values ( Y on N) to number 1 or 0

https://github.com/SaarthChahal/ML-DL/blob/main/cleaned%20dataset.png

**From shapes output: (985543, 23) i.e I was able to reduce 5803 records and 1 column.**

Data Plotting exercise to analyze relation ship between variables. Calculate the correlation matrix.   
There are too many variables to produce more readable correlation matrix and heatmap. Created 2 smaller array for matrix and heatmap for smoke and drink correlation 

Heatmap for smokers
https://github.com/SaarthChahal/ML-DL/blob/main/heatmap%20for%20smokers.png

Heatmap for drinkers
https://github.com/SaarthChahal/ML-DL/blob/main/heatmap%20for%20drinkers.png

Scatterplot for total cholestrol of smokers
https://github.com/SaarthChahal/ML-DL/blob/main/total%20cholestrol%20scatterplot%20for%20smokers.png

using sns.pairplot() created scatterplots between some of key variables
https://github.com/SaarthChahal/ML-DL/blob/main/scatterplots%20for%20pairs%20of%20variables.png






