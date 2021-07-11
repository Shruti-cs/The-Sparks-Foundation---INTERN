#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# 
# Intership Program
# 
# #GRIPJULY21
# 
# Task - 2: Prediction Using Unsupervised ML
# 
# Submitted by: Shruti.C.S

# In[1]:


# Importing Libraries required for Prediction:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[2]:


# Importing and Reading the Data as csv file:

irisdata_csv= "C:/SHRUTI/INTERNSHIP/THE SPARKS FOUNDATION/TASK-2/Iris.csv"
iris_data = pd.read_csv(irisdata_csv)
print("Data Imported Successfully!!!")
print("---------------------------------------------------")
print("Iris - Data:")
print("----------------------------------------------------")
iris_data


# In[3]:


#Checking the missing Values:

b=iris_data.isnull().values.any()
print("There is Null Value in Dataset: ",b)


# In[4]:


#Shape/ Size of the Data:

iris_data.shape


# In[5]:


#Information of the Data

iris_data.info()


# In[6]:


#Head of the Data:

iris_data.head()


# In[7]:


#Describing the Data:

iris_data.describe().T


# In[8]:


#Correlation of the Data:

iris_data.corr()


# In[9]:


#Grouping Data:

iris_data.groupby('Species').agg(["min","max","std","mean"])


# In[10]:


#Visualization of Data:
#Pairplot of the Data:

sns.pairplot(data=iris_data,hue="Species",kind="reg")
plt.show()


# In[12]:


#Combining Box & Strip Plots:

i_data=iris_data.drop([ 'Id','Species'] ,axis=1)
cols=i_data.columns
for i in cols:
    sns.boxplot(data=iris_data,x='Species',y=iris_data[i])
    sns.stripplot(data=iris_data,x='Species',y=iris_data[i])
    plt.show()


# In[13]:


#Heatmap correlation of the Data:

sns.heatmap(iris_data.corr(),cmap='summer', linewidths =4,linecolor='black' ,annot=True)


# In[14]:


#For each K value we Initialise k-means:

wcss =[]
k=range(1,9)

for i in k:
    Kmeans=KMeans(n_clusters=i)
    Kmeans.fit(i_data)
    wcss.append(Kmeans.inertia_)
    
#Elbow Method:


plt.plot(k,wcss,'bx-')
plt.xlabel("k")
plt.ylabel("Within Cluster Sum of Square")
plt.title("Elbow Method For Optimal k")
plt.show()

print("RESULT:\n\tFrom the Elbow Method Graph, We can conclude that '3' is optimum number of  clusters for k-Means Classification")


# In[15]:


# Creating the kmeans classifier:

x = i_data.iloc[:, [0, 1, 2, 3]].values
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[16]:


#Visualization of kmeans classifier:
#ScatterPlot:

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'gray', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'lime', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'red', label = 'Centroids')
plt.legend()


# In[17]:


#Analysis of Cluster :

i_data.index=pd.RangeIndex(len(i_data.index))
iris_km = pd.concat([i_data,pd.Series(kmeans.labels_)],axis=1)
iris_km.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','ClusterId']

slength=pd.DataFrame(iris_km.groupby(['ClusterId']).agg({'SepalLengthCm':'mean'}))
swidth=pd.DataFrame(iris_km.groupby(['ClusterId']).agg({'SepalWidthCm':'mean'}))
plength=pd.DataFrame(iris_km.groupby(['ClusterId']).agg({'PetalLengthCm':'mean'}))
pwidth=pd.DataFrame(iris_km.groupby(['ClusterId']).agg({'PetalWidthCm':'mean'}))

#Printing Results:

print("Range Index of Dataset:\n\t",i_data.index)
print("\n")
print("Concatination of Dataset & K-means:\n",iris_km )


# In[18]:


#Clusters of Dataset:

iris_data2=pd.concat([pd.Series([0,1,2]),slength,swidth,plength,pwidth],axis=1)
iris_data2.columns=['ClusterId','SepalLengthCm_Mean','SepalWidthCm_Mean','PetalLengthCm_Mean','PetalWidthCm_Mean']
iris_data2


# In[19]:


#Visualization of Clusters Found:
#Countplot:

sns.countplot(data=iris_km,x='ClusterId')
plt.show()


# In[21]:


#Pairplot :

sns.pairplot(data=iris_km)


# In[ ]:




