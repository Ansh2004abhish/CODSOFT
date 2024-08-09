#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd 
# In[2]:


df=pd.read_csv('creditcard.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(inplace=True)


# In[8]:


df['Class'].value_counts()


# In[9]:


not_fraud = df[df.Class == 0]
fraud = df[df.Class == 1]


# In[10]:


not_fraud.shape


# In[11]:


fraud.shape


# In[12]:


not_fraud.describe()


# In[13]:


fraud.describe()


# In[14]:


not_fraud_sample=not_fraud.sample(n=492)


# In[15]:


not_fraud_sample


# In[16]:


new_df=pd.concat([not_fraud_sample,fraud] , axis=0)


# In[17]:


new_df


# In[18]:


new_df['Class'].value_counts()


# In[19]:


new_df.groupby("Class").mean()


# In[20]:


from sklearn.model_selection import train_test_split

X = new_df.drop('Class', axis=1)
y = new_df['Class']


# In[22]:


X


# In[23]:


y


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y , random_state=2 )


# In[25]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[26]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[27]:


y_pred = logreg.predict(X_test)


# In[28]:


X_train_prediction=logreg.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,y_train)


# In[29]:


training_data_accuracy


# In[30]:


X_test_prediction = logreg.predict(X_test)

test_data_accuracy = accuracy_score(X_test_prediction,y_test)

print('Accuracy score on Test Data: ', test_data_accuracy)


# In[ ]:




