#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# 
# Intership Program
# 
# #GRIPJULY21
# 
# Task - 6: Prediction Using Decision Tree Algorithm
# 
# Submitted by: Shruti.C.S

# In[1]:


# Importing Libraries required for Prediction:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import sklearn.datasets as datasets
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz,DecisionTreeClassifier
import pydotplus
from IPython.display import Image 


# In[2]:


# Importing and Reading the Data as csv file:

iris=datasets.load_iris()
iris_csv= "C:/SHRUTI/INTERNSHIP/THE SPARKS FOUNDATION/TASK-6/Iris.csv"
iris_data = pd.read_csv(iris_csv)
print("Data Imported Successfully!!!")
print("---------------------------------------------------")
print("Iris - Data:")
print("----------------------------------------------------")
iris_data


# In[3]:


#Checking the missing Values:

n=iris_data.isnull().values.any()
print("There is Null Value in Dataset: ",n)


# In[4]:


#Information of the Data

iris_data.info()


# In[5]:


#Head of the Data:

iris_data.head()


# In[6]:


#Describing the Data:

iris_data.describe().T


# In[7]:


#Correlation of the Data:

iris_data.corr()


# In[8]:


#Heatmap correlation of the Data:

sns.heatmap(iris_data.corr(), linewidths =4,linecolor='black' ,annot=True)


# In[9]:


#Grouping Data:

iris_data.groupby('Species').agg(["min","max","std","mean"]).T


# In[10]:


#Data Points Count Value:

iris_data.Species.value_counts()


# In[11]:


#Visualization of Data:
#Combining ScatterPlots:

i_data=iris_data.drop([ 'Id','Species'] ,axis=1)
cols=i_data.columns
for i in cols:
    sns.set_style('whitegrid')
    sns.swarmplot(data=iris_data,x='Species',y=iris_data[i])
    plt.show()


# DECISION TREE ALGORITHM:

# In[12]:


#Classes of Data:

tn=iris.target_names
print('Classes to Predict: ',tn)
print('--------------------------------------------------------')

#Features of Data:

fn =iris.feature_names
print("Features: ",fn)


# In[13]:


#Extracting Attributes of Data:

att=iris.data

#Extracting Target of Data:

tar=iris.target
print('Target: \n',tar)


# In[18]:


#Defining Decision Tree:

i_df =pd.DataFrame(att, columns=fn)
tree_classifier=tree.DecisionTreeClassifier()
tree_classifier.fit(i_df,tar)


# In[19]:


#Visualization Of  Tree:

dot_data = StringIO()
print(dot_data)
export_graphviz(tree_classifier, out_file=dot_data, feature_names=iris.feature_names,
                filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
print(graph)
Image(graph.create_png())


# The Above Decision Tree is Without Testing Model (i.e) Using the Attributes and Target of Iris Dataset.

# In[23]:


#Model Training:

att_train,att_test,tar_train,tar_test=train_test_split(att,tar,test_size=0.1,random_state=1)
tree_classifier1=tree.DecisionTreeClassifier()
tree_classifier1.fit(att_train,tar_train)
print(tree_classifier1)
print('---------------------------------------------------------------------------------')
tar_pred=tree_classifier1.predict(att_test)
print(tar_pred)
print('---------------------------------------------------------------------------------')
print("Training Complete")


# In[24]:


#Comparing Actual vs Predicted Value:

iris_df = pd.DataFrame({'Actual Value': tar_test,'Predicted Value': tar_pred})
iris_df


# The Predicted Value of Iris Datset is Same that of Actual Value.
# Hence, 'Decision Tree Classifer Created'

# In[25]:


#Visualization Of  Tree:

dot_data = StringIO()
print(dot_data)
export_graphviz(tree_classifier1, out_file=dot_data, feature_names=iris.feature_names,
                filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
print(graph)
Image(graph.create_png())


# The Above Decision Tree is done with Testing Model of Iris Dataset.

# In[ ]:




