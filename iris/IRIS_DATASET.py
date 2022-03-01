#!/usr/bin/env python
# coding: utf-8

# ## Iris dataset 

# In[1]:


import numpy as np
import pandas as pd 
import pickle

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets


# In[2]:


#import data 
iris = datasets.load_iris()


# In[3]:


iris.feature_names


# In[4]:


iris


# In[5]:


#create a data frame
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["Species"]=iris.target


# In[6]:


df


# In[7]:


x=df.drop("Species",axis=1)
y=df["Species"]


# In[8]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=1)


# In[9]:


from sklearn.linear_model import  LogisticRegression
log_reg=LogisticRegression()


# In[10]:


log_reg.fit(x_train,y_train)




pickle.dump(log_reg, open('iris.pkl', 'wb'))


# In[11]:


train_predictions=log_reg.predict(x_train)
test_predictions=log_reg.predict(x_test)


# In[12]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[13]:


print("confusion matrix on train data ")
print(confusion_matrix(y_train,train_predictions))
print("\n")
print("accuracy score",accuracy_score(y_train,train_predictions))


# In[14]:


print("confusion matrix on test data ")
print(confusion_matrix(y_test,test_predictions))
print("\n")
print("accuracy score",accuracy_score(y_test,test_predictions))