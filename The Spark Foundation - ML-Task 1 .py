#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML
# 
# ● Predict the percentage of an student based on the no. of study hours.
# 
# ● This is a simple linear regression task as it involves just 2 variables.
# 
# ● What will be predicted score if a student studies for 9.25 hrs/ day?

# # Niharika Marwah

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import seaborn as sns 
import scipy
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#python scikit-learn library is used


# In[2]:


#read data file
data=pd.read_csv("http://bit.ly/w-data")
data
data.describe()


# In[3]:


data.head()


# In[4]:


y=data['Scores'] #dependent variable
x=data['Hours'] #independent variable


# In[5]:



#scatter plot to visualize better
plt.scatter(x,y)
plt.xlabel('Hours studied',fontsize=20)
plt.ylabel('Percenatge score',fontsize=20)
plt.show()


# In[6]:


X = data.iloc[:, :-1].values  #attributes-input- train data
y = data.iloc[:, 1].values   #values-output- test data


# In[7]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[8]:


#Linear Regression Algorithm
from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(X_train, y_train) 

print("Trained data")


# In[9]:


#plotting regression line
line = reg.coef_*X+reg.intercept_ #y=mx+c 

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line); #plots the best fit line
plt.show()


# # Prediction

# In[10]:


print(X_test) # Testing data - In Hours
y_pred = reg.predict(X_test) # Predicting the scores


# In[11]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[12]:


# to find score for 9.5 hrs studied per day
hours = [[9.25]] 
own_pred = reg.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[13]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

