#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


import tensorflow
import warnings
warnings.filterwarnings("ignore")


# In[3]:


from tensorflow.keras.datasets import mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[4]:


#reshape my X_train and X_test and we also need to convert y into one-hot vector
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
print(X_train.shape)
print(X_test.shape)


# In[5]:


from tensorflow.keras.utils import to_categorical


# In[6]:


#One hot encoding the target 
#convert class vectors to binary class matrix
print(y_train[10])
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)
print(y_train[10])


# In[7]:


#As part of data preprocessing we will normalize the data
print(X_train.max())
print(X_train.min())
X_train = X_train/255.0
X_test = X_test/255.0
print(X_train.max())
print(X_train.min())


# In[ ]:





# In[8]:


#Lets create the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import optimizers,regularizers


# In[ ]:





# In[9]:


model = Sequential()
#input layer
model.add(Dense(256,input_shape=(784,),activation='relu'))

#hidden layer1
model.add(Dense(256,activation='relu'))



#output layer
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])


# In[10]:


print(model.summary())


# In[11]:


model.evaluate(X_train,y_train)


# In[12]:


model.evaluate(X_test,y_test)


# In[13]:


model.fit(X_train,y_train,epochs=15,validation_data=(X_test,y_test))


# In[14]:


model.evaluate(X_train,y_train)


# In[15]:


model.evaluate(X_test,y_test)


# In[ ]:





# In[ ]:




