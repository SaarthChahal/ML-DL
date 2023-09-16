
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

**Data Plotting exercise** to analyze relation ship between variables. Calculate the correlation matrix.   
There are too many variables to produce more readable correlation matrix and heatmap. Created 2 smaller array for matrix and heatmap for smoke and drink correlation 

Heatmap for smokers
https://github.com/SaarthChahal/ML-DL/blob/main/heatmap%20for%20smokers.png

Heatmap for drinkers
https://github.com/SaarthChahal/ML-DL/blob/main/heatmap%20for%20drinkers.png

Scatterplot for total cholestrol of smokers
https://github.com/SaarthChahal/ML-DL/blob/main/total%20cholestrol%20scatterplot%20for%20smokers.png

using sns.pairplot() created scatterplots between some of key variables
https://github.com/SaarthChahal/ML-DL/blob/main/scatterplots%20for%20pairs%20of%20variables.png

**Model training module**

**Learning Model**
Train linear regression model  
We will need to first split up our data into an X1 array(cholesterol)  that contains the features to train on, 
And a y1 array(SMK_stat_type_cd) with the target variable, 
split up our data into an X2 array(Kidney function) that contains the features to train on, 
And a y2 array(DRK_YN)

Train test split. test split is 40 % train set is 60 % 

Loading the linear regression Model
prediction on Training data 

**Model evlauation.**  
Let's evaluate the model by checking out it's coefficients and how we can interpret them.

Learning model intercept 1 (smokers): -1.2726375396245835
Learning model intercept 2 (drinkers) : 0.4141733052412185
Coefficients for smokers: https://github.com/SaarthChahal/ML-DL/blob/main/coefficient.png
Coefficient for drinkers: https://github.com/SaarthChahal/ML-DL/blob/main/coefficient2.png

Interpreting the coefficient.
For every one unit change in smoke status there is negative impact on Cholestrol ( refelcted as negative)
and increase in  triglyceride and  hemoglobin which negatively affect the health indicator. 


**Prediction from Model**
Prediction scatterplot for smokers: https://github.com/SaarthChahal/ML-DL/blob/main/prediction%20scatterplot%20for%20smokers.png

Displot prediction for smokers: https://github.com/SaarthChahal/ML-DL/blob/main/displot%20method%20for%20smokers.png

Prediction scatterplot for drinkers: https://github.com/SaarthChahal/ML-DL/blob/main/scatterplot%20prediction%20for%20drinkers.png

Displot scatterplot for drinkers: https://github.com/SaarthChahal/ML-DL/blob/main/displot%20method%20for%20smokers.png


**Regression Evaluation Metrics**
Here are three common evaluation metrics for regression problems:

Mean Absolute Error** (MAE) is the mean of the absolute value of the errors: is the easiest to understand, because it's the average error.

Mean Squared Error** (MSE) is the mean of the squared errors: is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.

Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors: is even more popular than MSE, because RMSE is interpretable in the "y" units.


Regression Evaluation Metrics for smokers:

MAE:1 1.1094765424193946

MSE:1 1.8638480376557032

RMSE:1 1.3652281998463491


Regression Evaluation Metrics for drinkers:

MAE:2 0.4814403810202582

MSE:2 0.23847155682695156

RMSE:2 0.4883354961775271

List of variables used: https://github.com/SaarthChahal/ML-DL/blob/main/variables.png


**DECISION TREE AND RANDOM FOREST MODEL**


supervised, regression machine learning problem. It’s supervised because we have both the features (data on health parameters) and the targets (Smokers and Drinkers) that we want to predict


The reported averages include macro average (averaging the unweighted mean per label), weighted average (averaging the support-weighted mean per label), and sample average (only for multilabel classification). Micro average (averaging the total true positives, false negatives and false positives) is only shown for multi-label or multi-class with a subset of classes, because it corresponds to accuracy otherwise and would be the same for all metrics


**Classfication metrics.** 

The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label a negative sample as positive. 

The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

Fscore. The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.    

Smokers metrics for decision tree:

              precision    recall  f1-score   support

           1       0.77      0.75      0.76    179903
           2       0.31      0.32      0.32     52132
           3       0.40      0.41      0.40     63628

    accuracy                           0.60    295663
    macro avg      0.49      0.49      0.49    295663
    weighted avg   0.61      0.60      0.61    295663
   

Confusion Matrix.

Confusion matrix to evaluate the accuracy of a classification. 

Confusion matrix usage to evaluate the quality of the output of a classifier on the data. 

The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier. 

The higher the diagonal values of the confusion matrix the better, indicating many correct predictions.

