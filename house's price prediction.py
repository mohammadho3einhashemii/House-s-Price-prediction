#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()


# In[3]:


train.describe()


# In[4]:


missing_values = train.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)


# In[5]:


numeric_features = train.select_dtypes(include=["int64","float64"])
corr_matrix = numeric_features.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,cmap="coolwarm",annot=False)

corr_matrix["SalePrice"].sort_values(ascending=False)


# In[6]:


sns.histplot(train["SalePrice"] , kde=True)


# In[7]:


train['SalePrice'].skew()


# In[8]:


train['SalePrice'] = np.log1p(train['SalePrice'])  
sns.histplot(train['SalePrice'], kde=True)
plt.title("Distribution of SalePrice after log transform")
plt.show()


# In[9]:


mis = train.isnull().sum()
mis[mis > 0].sort_values(ascending = False)


# In[10]:


numerical_fit = [cul for cul in train.columns if train[cul].dtype in ["int64","float64"]]
numerical_fit.remove("Id")
numerical_fit.remove("SalePrice")


# In[11]:


x = train[numerical_fit]
y = train["SalePrice"]

x=x.fillna(x.median())


# In[12]:


from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(x,y,random_state=1)


# In[13]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

model = RandomForestRegressor(n_estimators=100,random_state=1)

model.fit(x_train,y_train)


# In[14]:


preds = model.predict(x_val)

#print("the accuracy is : ",accuracy_score(y_val,preds))
mae = mean_absolute_error(y_val,preds)
print("the mean absolute error is: ",mae) 


# In[15]:


x_test = test[numerical_fit]
x_test = x_test.fillna(x_test.median())


# In[16]:


test_preds = model.predict(x_test)

output = pd.DataFrame({"Id":test.Id , "SalePrice": test_preds})
output.to_csv('C:/Users/m/Desktop/submission.csv', index=False)


# In[ ]:




