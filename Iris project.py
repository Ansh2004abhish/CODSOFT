#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 


# In[2]:


df=pd.read_csv('IRIS.csv')


# In[3]:


df


# In[4]:


df.species.unique()


# In[5]:


df.iloc[66]


# In[6]:


df.isnull().sum()


# In[7]:


df.iloc[:,1]


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def knn(df,a,b,c,d,k):
    df=pd.DataFrame(df)
  #euclidian distances
    dist=[]
    for i in range(len(df)):
        dist.append(np.sqrt((((a - df.iloc[i,0]))**2)+(((b - df.iloc[i,1]))**2)+(((c - df.iloc[i,2]))**2)+(((d - df.iloc[i,3]))**2)))
    df["dist"]=dist
    df=df.sort_values(by="dist")
    plt.scatter(x=df.iloc[:,0],y=df.iloc[:,1], c=df.iloc[:, -2].apply(lambda x: {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}[x]))
    plt.scatter(x=a,y=b,color="black")
    plt.show()
    l=[]
    for i in range(k):
        l.append(df.iloc[i,-2])
    c1=l.count("Iris-setosa")
    c2=l.count("Iris-versicolor")
    c3=l.count("Iris-virginica")

#     for i in range(k):
#         print(i)
#         if(df.iloc[i,-1] == "Iris-setosa"):
#             c1+=1
#         elif(df.iloc[i,-1] == "Iris-versicolor"):
#             c2+=1
#         else:
#             c3+=1
#     print('values=', c1,c2,c3)
    if(c1>c2 and c1>c3):
        return 1
    elif(c2>c3):
        return 2
    else:
        return 3


import pandas as pd
import matplotlib.pyplot as plt

a,b=5.1,3.5	
c,d=1.4,0.2
k=7

res=knn(df,a,b,c,d,k)
if(res == 1):
    print("The flower is of Iris-setosa species")
elif(res == 2):
    print("The flower is of Iris-versicolor species")
else:
  print("The flower is of Iris-virginica species")


# In[ ]:




