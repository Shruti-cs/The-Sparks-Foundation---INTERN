#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# 
# Intership Program
# 
# #GRIPJULY21
# 
# Task - 3: Exploratory Data Analysis - Retail
# 
# Submitted by: Shruti.C.S

# In[76]:


# Importing Libraries required for Prediction:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Importing and Reading the Data as csv file:

retail_csv= "C:/SHRUTI/INTERNSHIP/THE SPARKS FOUNDATION/TASK-3/SampleSuperstore.csv"
retail_data = pd.read_csv(retail_csv)
print("Data Imported Successfully!!!")
print("---------------------------------------------------")
print("SuperStores - Data:")
print("----------------------------------------------------")
retail_data


# In[3]:


#Checking the missing Values:

b=retail_data.isnull().values.any()
print("There is Null Value in Dataset: ",b)


# In[4]:


#Information of the Data

retail_data.info()


# In[5]:


#Unique Data Count:

retail_data.nunique()


# In[6]:


#Describing the Data:

retail_data.describe().T


# In[7]:


#Grouping Data:

retail_data.groupby('State').agg(["min","max","std","mean"]).head()


# In[8]:


#Checking and Droping the Duplicates:

dup=retail_data.duplicated().sum()
print(dup)
print("-----------------------------")
print("Dropping the Duplicates:")
retail_data.drop_duplicates()


# In[11]:


#Dropping Irrelevant Column:

col=['Postal Code','Country']
retail_data1=retail_data.drop(columns=col,axis=1)


# In[12]:


#Correlation of the Data:

retail_data1.corr()


# In[13]:


#Heatmap correlation of the Data:

sns.heatmap(retail_data1.corr(),cmap='YlGnBu', linewidths =4,linecolor='black' ,annot=True)


# In[14]:


#Covariance of the Data:

retail_data1.cov()


# In[15]:


#Heatmap covariance of the Data:

plt.figure(figsize=(9,6))
sns.heatmap(retail_data1.cov(),cmap='YlGnBu', linewidths =4,linecolor='black' ,annot=True)


# In[16]:


#Visualization of Data:
#Pairplot of Data:

sns.pairplot(retail_data1)


# In[40]:


#CountPlot State-Wise:

plt.figure(figsize=(16,8))
sns.countplot(retail_data1['State'])
plt.xticks(rotation=90)
plt.show()


# In[41]:


#CountPlot Category-Wise:

plt.figure(figsize=(10,5))
sns.countplot(retail_data1['Category'])
plt.xticks(rotation=90)
plt.show()


# In[42]:


#CountPlot SubCategory-Wise:

plt.figure(figsize=(16,8))
sns.countplot(retail_data1['Sub-Category'])
plt.xticks(rotation=90)
plt.show()


# In[43]:


#CountPlot Ship Mode-Wise:

plt.figure(figsize=(16,8))
sns.countplot(retail_data1['Ship Mode'])
plt.xticks(rotation=90)
plt.show()


# In[44]:


#CountPlot Region-Wise:

plt.figure(figsize=(16,8))
sns.countplot(retail_data1['Region'])
plt.xticks(rotation=90)
plt.show()


# In[45]:


#CountPlot Segment-Wise:

plt.figure(figsize=(16,8))
sns.countplot(retail_data1['Segment'])
plt.xticks(rotation=90)
plt.show()


# In[61]:


#Finding Sales for Each state:

sales_state=retail_data1.groupby('State',as_index=False)['Sales'].sum()
sales_state.head()


# In[52]:


#Ploting Sales in Statewise:

plt.figure(figsize=(16,8))
plt.bar(sales_state['State'],sales_state['Sales'],color='gray')
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Sales')
plt.title("Statewise Sales")
plt.show()


# In[53]:


#Finding Sales for Each Category & Sub Category:

category_sales=retail_data1.groupby('Category',as_index=False)['Sales'].sum()
print(category_sales)
print('------------------')
subcat_sales=retail_data1.groupby('Sub-Category',as_index=False)['Sales'].sum()
print(subcat_sales)


# In[54]:


#Ploting Sales in Categorywise:

plt.figure(figsize=(16,8))
plt.bar(category_sales['Category'],category_sales['Sales'],color='orange')
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Sales')
plt.title("Categorywise Sales")
plt.show()


# In[55]:


#Ploting Sales in Sub-Category wise:

plt.figure(figsize=(16,8))
plt.bar(subcat_sales['Sub-Category'],subcat_sales['Sales'])
plt.xticks(rotation=90)
plt.xlabel('Sub-Category')
plt.ylabel('Sales')
plt.title("Sub-Category wise Sales")
plt.show()


# In[62]:


#Finding Sales for Each Region:

region_sales=retail_data1.groupby('Region',as_index=False)['Sales'].sum()
print(region_sales)


# In[65]:


#Ploting Sales in Region wise:

plt.figure(figsize=(16,8))
plt.bar(region_sales['Region'],region_sales['Sales'],color='lime')
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Sales')
plt.title("Region wise Sales")
plt.show()


# In[60]:


#Pofit & Loss Statewise:

pl_state=retail_data1.groupby('State',as_index=False)['Profit'].sum()
pl_state.head()


# In[59]:


#Ploting Profit & Loss  Statewise:

plt.figure(figsize=(16,8))
plt.bar(pl_state['State'],pl_state['Profit'],color='green')
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Profit & Loss')
plt.title("Statewise Profit & Loss")
plt.show()


# From the Above Graph We Can See That The Which State Having Profit & Which State Having Loss.
# 
# The State's in Heavy Loss : Texas, Ohio,Pennsylvania & Illinois
# The State's in  Loss : Oregon,Florida,Arizona,Tennessee,Colorado & North Carolina

# In[66]:


#Finding Profit & Loss for Each Region:

region_pl=retail_data1.groupby('Region',as_index=False)['Profit'].sum()
print(region_pl)


# In[72]:


#Ploting Profit & Loss  Regionwise:

plt.figure(figsize=(16,8))
plt.bar(region_pl['Region'],region_pl['Profit'],color='yellow')
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Profit & Loss')
plt.title("Regionwise Profit & Loss")
plt.show()


# From The Graph Above We Can See That 'East' & 'West' Regions are GAining More Profit. Whereas 'Central' Region is Gaining the Less ProfitCompared to Other Regions.

# In[83]:


#plotting:

plt.figure(figsize=(16,8))
sns.countplot(x='Category',hue='Region',data=retail_data1)


# In[84]:


#plotting:

plt.figure(figsize=(16,8))
sns.countplot(x='Sub-Category',hue='Region',data=retail_data1)


# In[108]:


#Region Analysis:

region=retail_data1.groupby(['Region'])[['Sales','Discount','Profit']].mean()
region


# In[119]:


#Line Graph on Profit , Sales & discount:

region.plot(kind='line',figsize=(15,10))
plt.title("Regionwise Analysis")
plt.xlabel('Region')
plt.ylabel('count')
plt.show()


# In[87]:


#State Analysis:

state=retail_data1.groupby(['State'])[['Sales','Discount','Profit']].mean()
state.head()


# In[104]:


#Line Graph on Profit , Sales & discount:

state.plot(kind='line',figsize=(15,10))
plt.title("Statewise Analysis")
plt.xlabel('State')
plt.ylabel('count')
plt.show()


# In[105]:


#City Analysis:

city=retail_data1.groupby(['City'])[['Sales','Discount','Profit']].mean()
city.head()


# In[110]:


#Line Graph on Profit , Sales & discount:

city.plot(kind='line',figsize=(15,10))
plt.title("Citywise Analysis")
plt.xlabel('City')
plt.ylabel('count')
plt.show()


# In[113]:


#Category Analysis:

category=retail_data1.groupby(['Category'])[['Sales','Discount','Profit']].mean()
category


# In[120]:


#plotting:

category.sort_values('Profit')[['Sales','Profit']].plot(kind='bar',figsize=(15,5))


# In[121]:


#Sub-Category Analysis:

s_category=retail_data1.groupby(['Sub-Category'])[['Sales','Discount','Profit']].mean()
s_category


# In[122]:


#plotting:

s_category.sort_values('Profit')[['Sales','Profit']].plot(kind='bar',figsize=(15,5))


# From this Analysis I'm Concluding that,
#            Through this analysis we can see that many states are facing loss than profit,  So we have  to Decrease the discount percentage in states having heavy loss.So that we can increase the profit in those states.
#            In Categories we have to concentrate on 'Office Supplies' because its Supply is Lesser than other categories.
#            In Sub-Categories we have to concentrate on 'Fasteners','Labels','Envelopes','Art' because Supplies of these less compared to other sub-catgories.
#            In Region we have to concentrate on 'Central' & 'South' to increase the supplies.
#            So by minimizing the discount rates and making more advertisements in various platforms will imporve in sales and  gain profit.
