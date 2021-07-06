#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# 
# Intership Program
# #GRIPJULY21
# 
# Task - 1: Prediction Using Supervised ML
# 
# Submitted by: Shruti.C.S
# 

# In[1]:


# Importing Libraries required for Prediction:

import pandas as pd
import numpy as np  
import seaborn as sns
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn import metrics  
import matplotlib.pyplot as plt  


# In[2]:


# Importing and Reading the Data as csv file:

stdata_csv= "C:/SHRUTI/INTERNSHIP/THE SPARKS FOUNDATION/TASK-1/student_scores - student_scores.csv"
stdata = pd.read_csv(stdata_csv)
print("Data Imported Successfully!!!")
print("---------------------------------------------------")
print("Study Hours and Scores of Students - Data:")
print("----------------------------------------------------")
stdata


# In[3]:


#Shape/ Size of the Data:

stdata.shape


# In[4]:


#Information of the Data

stdata.info()


# In[5]:


#Head of the Data:

stdata.head(10)


# In[6]:


#Describing the Data:

stdata.describe()


# In[7]:


#Correlation of the Data:

stdata.corr()


# In[8]:


# Plotting the Dataset:
#Box Plots:

stdata.plot(kind='box',subplots=True,layout=(2,2))
plt.show()


# In[10]:


#ScatterPlot of the Data:

sns.regplot(x='Hours',y='Scores',data=stdata,color='red')


# In[11]:


#Histogram  of the Data:

stdata.hist()
plt.show()


# In[12]:


#2D Plot:

stdata.plot(x='Hours', y='Scores', style='.')  
plt.title('Hours Studied vs Score Percentage')  
plt.xlabel('Hours')  
plt.ylabel('Score')  
plt.show()


# In[13]:


#Heatmap correlation of the Data:

sns.heatmap(stdata.corr(), cmap='coolwarm',linewidths =4,linecolor='black',cbar_kws={'orientation':'horizontal'} ,annot=True)


# In[14]:


#Checking the missing Values:

stdata.isnull().sum()


# In[15]:


#Attributes and Labels of the Data:

x = stdata.iloc[:, :-1].values  
y = stdata.iloc[:, 1].values  

print("x:\n",x)
print("------------")
print("y:\n",y)


# In[16]:


#Splitting the Data:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
print(x_train, x_test, y_train, y_test)


# In[17]:


#Training Dataset:

st_regressor = LinearRegression()  
st_regressor.fit(x_train, y_train) 
print(st_regressor)
print("-----------------------------")
print("Training complete!!")
print("-----------------------------")
print("TRAINING ACCURACY:",st_regressor.score(x_train, y_train))


# In[18]:


#Testing Dataset:

print(st_regressor)
print("-----------------------------")
print("Test complete!!")
print("-----------------------------")
print("TEst_STING ACCURACY:",st_regressor.score(x_test, y_test))


# In[19]:


# Plotting the regression line:

line = st_regressor.coef_*x+st_regressor.intercept_


# In[20]:


# Plotting  the Test Data:

plt.scatter(x, y,color='black')
plt.plot(x, line,color='red');
plt.show()


# In[21]:


#Predicting the Data:

# Testing data - In Hours:
print(x_test) 
print("----------------")

# Predicting the scores:
y_predict = st_regressor.predict(x_test) 
print(y_predict)


# In[22]:


# Comparing Actual Value vs Predicted Value of the Data:

st_df = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_predict})  
st_df


# In[23]:


#Data Visualization between Actual and Predicted Values:

x_axis=np.arange(len(st_df))
plt.bar(x_axis - 0.2,st_df['Actual value'],0.4,label='Actual value',color='red')
plt.bar(x_axis + 0.2,st_df['Predicted value'],0.4,label='Predicted value',color='blue')
plt.title("Actual value vs Predicted value")
plt.legend()
plt.show()


# In[24]:


#Testing the Original Test:

Hrs=9.25
t_score=st_regressor.predict([[Hrs]])
print("Number of Hours : {}".format(Hrs))
print("---------------------------------------")
print("Predicted Score : {}".format(t_score[0]))


# In[48]:


#Evaluating the Model:

mae=metrics.mean_absolute_error(y_test, y_predict)
print('Mean Absolute Error:',mae) 
print("------------------------------------------------------")
mse=metrics.mean_squared_error(y_test, y_predict)
print('Mean Squared Error:',mse) 
print("------------------------------------------------------")
r2=metrics.r2_score(y_test, y_predict)
print('R-Squared:',r2)
print("------------------------------------------------------")


# In[ ]:




