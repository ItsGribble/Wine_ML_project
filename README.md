# Wine_ML_project
CISB60 Intro to ML Project using a wine dataset - Jorge Medina
Dataset used - http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

The white wine data set from UCI is used for this project to apply several machine learning algorithms to try to predict the quality of wine. The best
machine learning algorithm was chosen by comparing testing metrics. The chosen model was then exported into a .pkl file and deployed using Flask. 
The user then fills out several fields and the results return as; Low quality wine or High qulaity wine.

Files:
app.py - - - - - - - - - - this file was a template and was modified to fit the needs of this project.
.css and .html - - - - - - both of these files were templates and modified to fit the needs of this project.
wine_model.py - - - - -  - contains the code that cleans the data set of outliers and uses a supervised Support Vector Machine algorithm to return the results.
White_Wine_Project.ipynb - contains all of the data exploration, data cleaning, training of different models and the whole workflow of this project.
