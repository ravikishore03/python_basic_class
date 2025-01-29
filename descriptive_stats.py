#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Create a pandas series of batsman1 scores
import pandas as pd
s1 = [20,10,37,46,30,29,31,19,60,45]
scores1 = pd.Series(s1)
scores1


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.boxplot(scores1, vert=False)


# In[14]:


s1 = [20,10,37,46,30,19,60,45,120,150]
scores1 = pd.Series(s1)
scores1


# In[16]:


import matplotlib.pyplot as plt
plt.boxplot(scores1, vert=False)


# In[20]:


df = pd.read_csv("Universities.csv")
print(df)
#plot box plot for SAT column
plt.boxplot(df["SAT"])


# In[ ]:




