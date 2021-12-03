#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[14]:


def get_data():
    df = pd.read_csv("ecommerce_data.csv")
    data = df.to_numpy()
    
    X = data[:, :-1] # consis is_mobile, n_products_viewed, visit_duration, is_returning_visitor, time_of_day 
                     # 5 variables in total
    Y = data[:, -1]
    
    
    # normalization
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:,1].std() # normalize is_mobile
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:,2].std() # normalize n_products_viewed
    
    
    N, D = X.shape # N should be 500 and D should be 6
    
    X2 = np.zeros((N, D+3))
    X2[:, 0:(D-1)] = X[:, 0:(D-1)] # copy every col excep time_of_day
    
    for n in range(N):
        t = int(X[n, D - 1])
        X2[n, t+D-1] = 1
        
    # alternative
#     Z = np.zeros((N,4))
#     Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    return X2, Y


# In[22]:


X2, Y = get_data()
X2.shape
# Y.shape


# In[27]:


def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1] # left with columns where the Y values are less than or equal to 1
    Y2 = Y[Y <= 1]
    return X2, Y2


# In[28]:


X, Y = get_binary_data()


# In[29]:


X.shape
Y.shape


# In[30]:


D = X.shape[1] # the col number


# In[31]:


W = np.random.randn(D) # random the weight


# In[32]:


b = 0


# In[33]:


def sigmoid(a):
    return 1 / ( 1 + np.exp(-a))

def prediction(X, W, b):
    return np.round(sigmoid(X.dot(W) + b))

def classification_rate(Y, P):
    return np.mean(Y == P)


# In[34]:


predictions = prediction(X, W, b)


# In[35]:


classification_rate(Y, predictions)


# In[ ]:




