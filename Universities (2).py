#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[8]:


#Mean value of SAT scorE
df = pd.read_csv("Universities.csv")
df


# In[10]:


#np.mean(df["SAT"])


# In[12]:


np.median(df["SAT"])


# In[14]:


df.describe()


# In[16]:


#Standard deviation
np.std(df["GradRate"])


# In[20]:


#Find the variance
np.var(df["SFRatio"])


# In[22]:


df.describe()


# In[28]:


#visualize the GradRate using histogram
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[38]:


sns.histplot(df["Accept"])


# In[40]:


sns.histplot(df["Accept"], kde = True)


# ### Observation
# In Acceptance ratio the data distribution is non-symmetical and right skewed

# In[ ]:




