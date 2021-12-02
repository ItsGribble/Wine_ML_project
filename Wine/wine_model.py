import pandas as pd
import numpy as np
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm
import joblib

#import data set
w_wine = pd.read_csv('winequality-white.csv',sep=';')


#clean data
#use .quantile 95% to remove outliers above 2 standard deviations
#get the lower and upper limits of residual sugar
res_sug_min, res_sug_max = w_wine['residual sugar'].quantile([.05,.95])
#create a new dataframe without the outliers
w_wine1 = w_wine.loc[(w_wine['residual sugar']>res_sug_min) & (w_wine['residual sugar']<res_sug_max)].copy()

#so2 = sulfur dioxide
so2_min,so2_max = w_wine1['total sulfur dioxide'].quantile([.05,.95])
w_wine2 = w_wine1.loc[(w_wine1['total sulfur dioxide']>so2_min) & (w_wine1['total sulfur dioxide']<so2_max)].copy()

#grab the min and max threshold for chlorides using .quantile(.95)
chl_min,chl_max = w_wine2['chlorides'].quantile([.05,.95])
#create a third dataframe to remove outliers for chloride
w_wine3 = w_wine2.loc[(w_wine2['chlorides']>chl_min) & (w_wine2['chlorides']<chl_max)].copy()

#use .quantile(.95) to remove anything above 2 standard deviations
FA_min,FA_max = w_wine3['fixed acidity'].quantile([.05,.95])
w_wine4 = w_wine3.loc[(w_wine3['fixed acidity']>FA_min) & (w_wine3['fixed acidity']<FA_max)].copy()



#add a new column with the rating
w_wine4['rating'] = np.where(w_wine4['quality']>6,1,0)



#split the data
X = w_wine4.drop(['rating','quality'],axis=1)
y = w_wine4.rating

#train test spllit
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.3,random_state=101 )

#scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#use svc and set C to 20
final_model = SVC(C=20,)
#fit the model
final_model.fit(X_train,y_train)

#pickle dump
pickle.dump(final_model,open('wine_svc.pkl','wb'))

#load the model
load_model=joblib.load('wine_svc.pkl')
load_model