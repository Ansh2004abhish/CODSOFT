#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


advertising=pd.read_csv('advertising.csv')


# In[3]:


advertising.head()


# In[4]:


advertising.shape


# In[5]:


advertising.info()


# In[6]:


advertising.describe()


# In[7]:


advertising.isnull().sum()*100/advertising.shape[0]


# In[8]:


fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
plt.tight_layout()


# In[9]:


sns.boxplot(advertising['Sales'])
plt.show()


# In[11]:


sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[12]:


sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[13]:


X = advertising['TV']
y = advertising['Sales']


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[15]:


X_train.head()


# In[16]:


y_train.head()


# In[17]:


import statsmodels.api as sm


# In[18]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()


# In[19]:


lr.params


# In[20]:


print(lr.summary())


# In[21]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[22]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[23]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         
plt.show()


# In[24]:


plt.scatter(X_train,res)
plt.show()


# In[25]:



X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)


# In[26]:


y_pred.head()


# In[27]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[28]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[29]:


r_squared = r2_score(y_test, y_pred)
r_squared


# In[30]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# In[ ]:




