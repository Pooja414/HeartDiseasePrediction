#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[2]:


a=pd.read_csv(r"C:\Users\1pooj\Downloads\archive (7)\heart.csv")


# In[3]:


a.head()


# In[4]:


a.tail()


# In[5]:


a.shape


# In[6]:


a.isnull().sum()


# In[7]:


x=a.drop(columns='target',axis=1)


# In[8]:


y=a[['target']]


# In[9]:


x


# In[10]:


y


# In[11]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[12]:


print(x.shape,xtrain.shape,xtest.shape)


# In[13]:


model=LogisticRegression()
model.fit(xtrain,ytrain)


# In[14]:


xtrain_predict=model.predict(xtrain)
train_accuracy_score=accuracy_score(xtrain_predict,ytrain)


# In[15]:


train_accuracy_score


# In[16]:


xtest_predict=model.predict(xtest)
test_accuracy_score=accuracy_score(xtest_predict,ytest)


# In[17]:


test_accuracy_score


# In[18]:


input_data=(70,1,0,145,174,0,1,125,1,2.6,0,0,3)

input_data_as_numpy_array=np.array(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)

if(prediction[0]==0):
    print("person does not have heart disease")
    
else:
    print("person have heart disease")


# In[ ]:





# In[ ]:




