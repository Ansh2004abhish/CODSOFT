#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


MovieData=pd.read_csv('IMDb Movies India.csv')


# In[16]:


MovieData.head(10)


# In[17]:


MovieData.info()


# In[18]:


MovieData.shape


# In[20]:


MovieData.columns


# In[21]:


MovieData = MovieData.drop(columns=['Name', 'Actor 2', 'Actor 3'])
MovieData.head()


# In[22]:


MovieData.dropna(inplace=True)


# In[23]:


MovieData.drop_duplicates(inplace=True)
MovieData.shape


# In[24]:


MovieData['Year'].unique()


# In[25]:


def handleYear(value):
    value = str(value).strip('()')
    return int(value)
MovieData['Year'] = MovieData['Year'].apply(handleYear)
MovieData['Year'].head()


# In[26]:


MovieData['Duration'].unique()


# In[27]:


def handleDuration(value):
    value=str(value).split(' ')
    value=value[0]
    return int(value)
MovieData['Duration']=MovieData['Duration'].apply(handleDuration)
MovieData['Duration'].head()


# In[28]:


MovieData['Genre'].unique()


# In[29]:


def split_genre_column(MovieData,Genre):
   
    MovieData['Genre1'] = MovieData[Genre].str.split(',', expand=True)[0]
    MovieData['Genre2'] = MovieData[Genre].str.split(',', expand=True)[1]
    MovieData['Genre3'] = MovieData[Genre].str.split(',', expand=True)[2]
    return MovieData

split_genre_column(MovieData,'Genre')


# In[30]:


MovieData.isna().sum()


# In[31]:


MovieData = MovieData.fillna(0)
MovieData.isna().sum()


# In[32]:


G=['Genre1','Genre2','Genre3']
for x in G:
    MovieData[x],_ = pd.factorize(MovieData[x])
    
MovieData = MovieData.drop(columns=['Genre'])
MovieData.head(3)


# In[33]:


MovieData['Votes'].unique()


# In[34]:


def handleVotes(value):
    value = str(value).replace(',','')
    return int(value)
MovieData['Votes'] = MovieData['Votes'].apply(handleVotes)
MovieData['Votes'].head()


# In[35]:


MovieData['MovieAge'] = 2024 - MovieData['Year']
MovieData['MovieAge'] 


# In[36]:


DirectorCounts =MovieData['Director'].value_counts()
MovieData['DirectorPopularity']= MovieData['Director'].map(DirectorCounts)
ActorCounts= MovieData['Actor 1'].value_counts() 
MovieData['ActorPopularity']=MovieData['Actor 1'].map(ActorCounts) 


# In[37]:


MovieData['LogVotes']=np.log1p(MovieData['Votes'])
MovieData['LogVotes'] 


# In[38]:


DirectorAvgRating = MovieData.groupby('Director')['Rating'].mean()
MovieData['DirectorAvgRating'] = MovieData['Director'].map(DirectorAvgRating)

ActorAvgRating = MovieData[['Actor 1']].stack().reset_index(name='Actor')
ActorAvgRating = ActorAvgRating.merge(MovieData[['Rating']], left_on='level_0', right_index=True)
ActorAvgRating = ActorAvgRating.groupby('Actor')['Rating'].mean()
MovieData['ActorAvgRating'] = MovieData['Actor 1'].map(ActorAvgRating) 


# In[39]:


plt.figure(figsize=(12, 8))
TopDirectors=MovieData['Director'].value_counts().index[0:10]
sns.boxplot(data=MovieData[MovieData['Director'].isin(TopDirectors)], x='Director', y='Rating', palette='rocket')

plt.title('Box Plot of Ratings by Top Directors')
plt.xlabel('Director')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()


# In[40]:


plt.figure(figsize=(12, 8))
TopActors = MovieData['Actor 1'].value_counts().index[:10]
sns.boxplot(data=MovieData[MovieData['Actor 1'].isin(TopActors)], x='Actor 1', y='Rating', palette='mako')

plt.title('Box Plot of Ratings by Top Actors')
plt.xlabel('Actor 1')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()


# In[41]:


GenreColumns=['Genre1','Genre2','Genre3','Rating']
plt.figure(figsize=(15,12))
sns.pairplot(MovieData[GenreColumns],diag_kind='kde',kind='scatter')

plt.suptitle('Pairplot of Factorized Genres and Ratings', y=1.02)
plt.show()


# In[42]:


plt.figure(figsize=(10,6))
sns.histplot(MovieData['Duration'],bins=20,kde=True,color='lightcoral')

plt.title('Distribution of Movie Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.show()


# In[43]:


plt.figure(figsize=(25,9))
sns.boxplot(data=MovieData,x='Year',y='Rating',palette='Spectral')
sns.dark_palette("#69d",reverse=True,as_cmap=True)
plt.title('Box Plot of Ratings by Year')
plt.xlabel('Year')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()


# In[44]:


plt.figure(figsize=(10,6))
sns.histplot(MovieData['Votes'],bins=20,kde=True,color='blue')
plt.title('Distribution of Movie Votes')
plt.xlabel('Votes')
plt.ylabel('Frequency')
plt.xscale('log')
plt.show()


# In[45]:


plt.figure(figsize=(17,8))
sns.violinplot(data=MovieData,x='Genre1',y='Rating',palette='Set2')
plt.title('Violin Plot of Ratings by Genre')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()


# In[46]:


sns.jointplot(data=MovieData,x='Votes',y='Rating',kind='hex',cmap='plasma')
plt.suptitle('Joint Plot of Votes vs Rating', y=1.02)
plt.show()


# In[47]:


MovieData['Genre1 encoded'] =round(MovieData.groupby('Genre1')['Rating'].transform('mean'),1)
MovieData['Genre2 encoded']=round(MovieData.groupby('Genre2')['Rating'].transform('mean'),1)
MovieData['Genre3 encoded'] =round(MovieData.groupby('Genre3')['Rating'].transform('mean'),1)
MovieData['Votes encoded'] =round(MovieData.groupby('Votes')['Rating'].transform('mean'), 1)
MovieData['Director encoded']= round(MovieData.groupby('Director')['Rating'].transform('mean'), 1)
MovieData['Actor 1 encoded']= round(MovieData.groupby('Actor 1')['Rating'].transform('mean'), 1)
MovieData.head()


# In[48]:


MovieData.drop(columns=['Genre1','Votes','Director','Actor 1','Genre2','Genre3'],inplace=True)
MovieData['Rating'] =round(MovieData['Rating'],1)


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


# In[52]:


X=MovieData.drop("Rating",axis=1)
Y=MovieData["Rating"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=52)

model=LinearRegression()
model.fit(X_train,Y_train)

X_test_prediction= model.predict(X_test)

mse =mean_squared_error(Y_test,X_test_prediction)
print(f"Mean Squared Error (MSE): {mse:.2f}")

r2 = r2_score(Y_test,X_test_prediction)
print(f"R-squared score: {r2:.2f}")


# In[53]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)

model_DT=DecisionTreeRegressor(random_state=42)
model_DT.fit(X_train,Y_train)
X_test_prediction_DT=model_DT.predict(X_test)

mse_DT =mean_squared_error(Y_test,X_test_prediction_DT)
print(f"Mean Squared Error (MSE): {mse_DT:.2f}")

r2_DT = r2_score(Y_test,X_test_prediction_DT)
print(f"R-squared score: {r2_DT:.2f}")


# In[54]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)

model_DT=DecisionTreeRegressor(random_state=42)
model_DT.fit(X_train,Y_train)
X_test_prediction_DT=model_DT.predict(X_test)

mse_DT =mean_squared_error(Y_test,X_test_prediction_DT)
print(f"Mean Squared Error (MSE): {mse_DT:.2f}")

r2_DT = r2_score(Y_test,X_test_prediction_DT)
print(f"R-squared score: {r2_DT:.2f}")


# In[55]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)

model_SVR = SVR(kernel='linear',C=1.0,epsilon=0.1,gamma='scale')
model_SVR.fit(X_train,Y_train)
X_test_prediction_SVR=model_SVR.predict(X_test)

mse_SVR =mean_squared_error(Y_test,X_test_prediction_SVR)
print(f"Mean Squared Error (MSE): {mse_SVR:.2f}")

r2_SVR= r2_score(Y_test,X_test_prediction_SVR)
print(f"R-squared score: {r2_SVR:.2f}")


# In[56]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)

model_NN = MLPRegressor(hidden_layer_sizes=(100,),random_state=42)
model_NN.fit(X_train,Y_train)
X_test_prediction_NN=model_NN.predict(X_test)

mse_NN =mean_squared_error(Y_test,X_test_prediction_NN)
print(f"Mean Squared Error (MSE):{mse_NN:.2f}")

r2_NN= r2_score(Y_test,X_test_prediction_NN)
print(f"R-squared score:{r2_NN:.2f}")


# In[61]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)

model_GB = GradientBoostingRegressor(n_estimators=100,random_state=50)
model_GB.fit(X_train,Y_train)
X_test_prediction_GB=model_GB.predict(X_test)

mse_GB =mean_squared_error(Y_test,X_test_prediction_GB)
print(f"Mean Squared Error (MSE): {mse_GB:.2f}")

r2_GB= r2_score(Y_test,X_test_prediction_GB)
print(f"R-squared score: {r2_GB:.2f}")


# In[ ]:




