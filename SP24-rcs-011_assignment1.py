#!/usr/bin/env python
# coding: utf-8

# In[2]:


nums = [3,5,7,8,12]
cubes = []
for number in nums:
    cubes.append(number ** 3)
print(cubes)


# In[5]:


myDict = {}

myDict['parrot'] = 2
myDict['goat'] = 4
myDict['spider'] = 8
myDict['crab'] = 10

print(myDict)


# In[6]:


myDict = {'parrot': 2, 'goat': 4, 'spider': 8, 'crab': 10}

total_legs = 0

for animal, legs in myDict.items():
    print(f"The {animal} has {legs} legs.")
    total_legs += legs 
print(f"Total number of legs: {total_legs}")


# In[7]:


A = (3, 9, 4, [5, 6])
A[3][0] = 8
print(A)


# In[8]:


del A


# In[9]:


B = ('a', 'p', 'p', 'l', 'e')
count_p = B.count('p')
print(f"The number of occurrences of 'p': {count_p}")


# In[10]:


index_l = B.index('l')
print(f"The index of 'l': {index_l}")


# In[11]:


import numpy


# In[12]:


import numpy as np

A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print(A)


# In[34]:


import numpy as np

A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

B = A[:2, :2]
print (B)


# In[15]:


import numpy as np
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
C = np.empty_like(A)

print(C)


# In[16]:


import numpy as np
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
z = np.array([1, 0, 1])

C = np.empty_like(A)

for i in range(A.shape[1]):  
    C[:, i] = A[:, i] + z

print(C)


# In[17]:


X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])
v = np.array([9, 10])


# In[18]:


import numpy as np

X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

result = X + Y
print(result)


# In[20]:


import numpy as np

X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

multi = np.dot(X, Y)

print(multi)


# In[21]:


import numpy as np

Y = np.array([[5, 6], [7, 8]])

sqrt_Y = np.sqrt(Y)

print(sqrt_Y)


# In[22]:


import numpy as np
X = np.array([[1, 2], [3, 4]])
v = np.array([9, 10])

result = np.dot(X, v)

print(result)


# In[23]:


import numpy as np

X = np.array([[1, 2], [3, 4]])

column_sums = np.sum(X, axis=0)

print(column_sums)


# In[39]:


def Compute(time,distance):

    if time == 0:
        raise error("time cannot be zero")
    velocity = time / distance
    return velocity
distance = 80  
time = 20       

velocity = Compute(distance, time)
print(f"The velocity is {velocity} meters per second.")


# In[38]:


even_num = [2, 4, 6, 8, 10, 12]

def mult(even_num):
    product = 1  
    
    for num in even_num:
        product *= num
    
    return product

result = mult(even_num)
print(f"The product of all even numbers in the list is: {result}")


# In[40]:


import pandas as pd

data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

df = pd.DataFrame(data)

print(df.head(2))


# In[41]:


import pandas as pd

data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

df = pd.DataFrame(data)

print(df['C2'])


# In[42]:


import pandas as pd

data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

df = pd.DataFrame(data)

df = df.rename(columns={'C3': 'B3'})

print(df)


# In[43]:


import pandas as pd
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'B3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

df = pd.DataFrame(data)

df['Sum'] = df.sum(axis=1)

print(df)


# In[10]:


import pandas as pd

data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'B3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

df = pd.DataFrame(data)

df['Sum'] = df.sum(axis=1)

print(df)


# In[20]:


print("DataFrame loaded from 'hello_sample.csv':")
print(df)


# In[19]:


print(df.tail(2))


# In[18]:



print(df.info())


# In[17]:



print(df.shape)


# In[16]:


sorted_df = df.sort_values(by='Weight')

print(sorted_df)


# In[21]:


print("Missing values in the DataFrame (True indicates missing values):")
print(df.isnull())

df_dropped_rows = df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(df_dropped_rows)

df_dropped_columns = df.dropna(axis=1)
print("\nDataFrame after dropping columns with missing values:")
print(df_dropped_columns)


# In[ ]:




