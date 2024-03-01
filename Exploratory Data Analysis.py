#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('house_data.zip')


# In[3]:


df


# # Data Wrangling (Data preparation)

# In[4]:


df.isnull().sum().tail(41)


# In[5]:


df.isnull().sum().head(41)


# In[6]:


df.drop(['Pool QC','Fence','Misc Val','Alley','Mas Vnr Type'],axis=1,inplace=True)


# In[7]:


df.shape


# In[8]:


cols=df.select_dtypes(exclude='number')


# In[9]:


len(cols.columns)


# In[ ]:





# In[10]:


df.info()


# In[11]:


num_cols=df.describe()
num_cols


# # Line Plot
# * Relationship plot(Bivariate)
# * Trends/values- univariate)

# In[12]:


#Univariate graph
df['SalePrice'].plot(figsize=(15,8))


# Analysis- we analysed in this project that the price of mostly houses are nearly 2 lakhs.

# In[6]:


sns.set_style('whitegrid')
plt.figure(figsize=(15,8))
plt.plot(df['SalePrice'],color='green',alpha=0.5)
plt.title('Price of Vehicle Line Plot',size=30,color='red')
plt.xlabel('Row Indexes',color='green',size=20)
plt.ylabel('Sale Price',color='green',size=20)
plt.show()


# In[8]:


#Bivariate graph
sns.set_style('whitegrid')
plt.figure(figsize=(15,9))
plt.plot(df['Yr Sold'],df['SalePrice'],color='deeppink',alpha=0.4)
plt.title('Line Graph between Sale Price & Yr Sold',color='red')
plt.xlabel('Yr Sold',color='green',size= 20)
plt.ylabel('SalePrice',color='green',size=20)
plt.show()         


# In[ ]:


Analysis= max price house is sold in 2007.


# # Bar Plot
# * Bivariate analysis
# * Bar chart used by Categorical, Nominal, Categiical vs Numerical
# * Bar plots are used to display counts of unique value of categorical dtypes, height of bar represents counts for each category.

# In[14]:


sns.set_style('whitegrid')
df['SalePrice'].value_counts().plot.bar(figsize=(25,8))
plt.xlabel('Sale Price',size='20',color='red')


# In[18]:


max(df['SalePrice'])


# In[20]:


df['SalePrice'].value_counts()


# In[22]:


df['Yr Sold'].value_counts()


# In[38]:


sns.set_style('whitegrid')
df['Yr Sold'].value_counts().plot.bar(figsize=(15,8))
plt.xlabel('Year',size=20,color='green')
plt.ylabel('No of house sold',size=20,color='green')
plt.title('Yearly House Sold',color='red',size=25)
plt.show()


# In[39]:


df.head()


# In[41]:


df[['Lot Frontage','Lot Area']][10:50].plot.bar(figsize=(15,8))


# In[44]:


df[['MS SubClass','Mo Sold']].value_counts().plot.bar(figsize=(15,8))


# # Histogram
# * Continuous Samples- study the spread/ distribution of data.
# * Univariate analysis

# In[52]:


plt.figure(figsize=(20,6))
plt.hist(df['SalePrice'],color='lightgreen',bins=15)
plt.title('Histogram plot of SalePrice',color='red',size=25)
plt.xlabel('SalePrice',color='green',size=20)
plt.ylabel('frequency',color='green',size=20)
plt.show()


# In[51]:


df['SalePrice'].value_counts()


# ## Subplots

# In[65]:


df['Sale Condition'].value_counts()


# In[68]:


#Subplots=(1,2,1)-- 1st Row, 2nd Row, 3rd Row
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.hist(df['SalePrice'],color='lightgreen',bins=15)
plt.title('Histogram plot of SalePrice',color='red',size=25)
plt.xlabel('SalePrice',color='green',size=20)
plt.ylabel('frequency',color='green',size=20)
plt.show()

plt.subplot(1,2,2)
plt.hist(df['Sale Condition'],color='lightgreen',bins=15)
plt.title('Histogram plot of Sale Condition',color='red',size=25)
plt.xlabel('Sale Condition',color='green',size=20, style='italic')
plt.ylabel('frequency',color='green',size=20)
plt.show()


# In[58]:


df['Yr Sold'].value_counts()


# In[27]:


# Histogram with loop
num_cols=['MS SubClass','Lot Frontage','Lot Area','Mo Sold']
for i in range (0,len(num_cols),2):
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    sns.distplot(df[num_cols[i]],kde=False)   #kde=kernal density function
    plt.subplot(122)
    sns.distplot(df[num_cols[i+1]],kde=False)
    plt.tight_layout()
    plt.show()


# In[28]:


# Histogram with loop
num_cols=['MS SubClass','Lot Frontage','Lot Area','Mo Sold']
for i in range (0,len(num_cols),2):
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    sns.distplot(df[num_cols[i]],kde=True)   #kde=kernal density function
    plt.subplot(122)
    sns.distplot(df[num_cols[i+1]],kde=True)
    plt.tight_layout()
    plt.show()


# In[12]:


df.head()


# # Scatter Plot
# * Scatterplot in plotted between numerical values.
# * shows relationdhip between x & y plots

# In[62]:


plt.figure(figsize=(10,7))
sns.scatterplot(x=df['MS SubClass'],y=df['Sale Condition'],color='crimson',marker='o',alpha=0.7)
plt.show()


# In[ ]:


max(df['Lot Area']


# In[77]:


plt.figure(figsize=(10,7))
sns.scatterplot(x=df['Street'],y=df['Lot Area'],hue=df['SalePrice'],color=/0258)
plt.show()


# # Box plot
# * Distribution of sample
# * Five point summary- min,max,q1,median,q3
# * Outliers

# In[49]:


for column in num_cols:
    plt.figure(figsize=(12,2))
    sns.boxplot(data=num_cols,x=column,color='green')
    plt.title('Boxplot for numerical column',color='red')
    plt.show()
    


# In[78]:


plt.figure(figsize=(10,5))
sns.boxplot(x='SalePrice',data=df)


# In[53]:


plt.figure(figsize=(20,8))
sns.boxplot(x='Sale Condition',y='SalePrice',data=df)
plt.xlabel('Sale Condition',color='blue',size=20)
plt.ylabel('Sale Price',color='blue',size=20)
plt.title('Boxplot for columns',size=25,color='red')


# 
# # Distribution Plot
# * Histogram + Density
# * Probability Density Function

# In[81]:


plt.figure(figsize=(10,6))
sns.distplot(df['SalePrice'],color='crimson',hist=False)
plt.show()


# # Pie Chart
# * Distribution in percentage(%)

# In[86]:


plt.figure(figsize=(10,10))
plt.pie(df['Sale Condition'].value_counts(),autopct='%0.2f%%',labels=df['Sale Condition'].value_counts().index)
plt.show()


# In[88]:


df['Sale Condition'].value_counts().index


# In[70]:


df.head()


# # Heatmap
# * Multivariate data

# In[21]:


df.loc[:,'Misc Val':'Yr Sold']


# In[23]:


table.corr()


# In[27]:


sns.heatmap(table.corr(),annot=True,cmap= 'magma')
plt.show()


# # CountPlot

# In[8]:


plt.figure(figsize=(12,6))
sns.countplot(x='Sale Condition',data=df)


# # Multivariate Analysis
# * Numeric vs Numeric
# * Numeric vs Categorical
# * Categorical vs Categirical

# ## Numeric vs Numeric

# In[14]:


sns.jointplot(data=df,x='MS SubClass',y='Mo Sold')


# ## Categorical Vs Categorical

# In[19]:


sns.jointplot(data=df,x='MS SubClass',y='Mo Sold',hue='SalePrice')


# In[6]:


sns.jointplot(data=df,x='MS SubClass',y='Mo Sold',hue='Sale Condition',kind='kde')


# In[10]:


sns.jointplot(data=df,x='MS SubClass',y='Mo Sold', kind='reg')


# In[11]:


sns.jointplot(data=df,x='MS SubClass',y='Mo Sold',kind='hist')


# In[12]:


sns.jointplot(data=df,x='MS SubClass',y='Mo Sold',kind='hex')


# # Violinplot

# In[13]:


sns.violinplot(x=df['SalePrice'])


# In[14]:


sns.violinplot(data=df,x='SalePrice',y='MS SubClass')


# In[6]:


df.head(2)


# In[9]:


df['Overall Cond']


# In[12]:


sns.swarmplot(x='Overall Cond',y='SalePrice',data=df,hue='Overall Cond',palette='magma')
plt.show()

