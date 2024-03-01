#!/usr/bin/env python
# coding: utf-8

# # Customer segmentation using KMeans clustering by KODI VENU

# In[1]:


import pandas as pd
data=pd.read_csv('Mall_Customers.csv')


# Top 5 rows

# In[2]:


data.head()


# Last 5 rows

# In[3]:


data.tail()


# Dataset Shape

# In[4]:


data.shape


# In[5]:


print('Number of rows', data.shape[0])
print('Number of columns', data.shape[1])


# Dataset Information

# In[6]:


data.info()


# Check null values in the dataset

# In[7]:


data.isnull().sum()


# Dataset Statistics

# In[8]:


data.describe()


# KMeans clustering

# In[9]:


data.columns


# In[11]:


X=data[['Annual Income (k$)','Annual Income (k$)']]
X


# In[15]:


import warnings
warnings.filterwarnings('ignore')


# In[16]:


from sklearn.cluster import KMeans
k_means=KMeans()
k_means.fit(X)
k_means.fit_predict(X)


# Elbow method to find optimal number of clusters

# In[17]:


wcss=[]
for i in range(1,11):
    k_means=KMeans(n_clusters=i)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
wcss


# In[18]:


import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# Model training

# In[25]:


data.columns


# In[26]:


X=data[['Annual Income (k$)','Spending Score (1-100)']]


# In[27]:


k_means=KMeans(n_clusters=5,random_state=42)
y_means=k_means.fit_predict(X)


# In[28]:


y_means


# In[29]:


data.columns


# In[30]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='red',label="cluster1")
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='yellow',label="cluster2")
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='green',label="cluster3")
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='blue',label="cluster4")
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='black',label="cluster5")
plt.legend()
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100,c="magenta")
plt.title("customer segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()


# In[31]:


k_means.predict([[15,39]])


# Save the model

# In[33]:


import joblib
joblib.dump(k_means,"customer_segmentation")
model=joblib.load("customer_segmentation")


# In[34]:


model


# In[36]:


model.predict([[15,39]])


# In[ ]:




