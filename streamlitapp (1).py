#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np


# In[114]:


data = pd.read_csv('House_price_prediction.csv')


# In[22]:


data.drop(columns=['area_type','availability','society'],inplace=True)


# In[24]:


data.dropna(inplace=True)


# In[26]:


data.location.value_counts()


# In[28]:


data['location'] = data['location'].apply(lambda x:x.strip())


# In[36]:


data.location.value_counts()


# In[32]:


location_stats = data.groupby('location')['location'].agg('count').sort_values(ascending=False)


# In[33]:


location_less_than_10_entries = location_stats[location_stats <=10]


# In[34]:


location_less_than_10_entries


# In[37]:


data['location'] = data['location'].apply(lambda x:'other' if x in location_less_than_10_entries else x)


# In[39]:


data['location'].value_counts()


# In[40]:


data


# In[41]:


data['size'].unique()


# In[43]:


data['bedrooms'] = data['size'].apply(lambda x:int(x.split(' ')[0]))


# In[44]:


data


# In[45]:


data.total_sqft.unique()


# In[46]:


def clean(sqft):
    tokens = sqft.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    else:
        try:
            return float(sqft)
        except:
            return None




# In[49]:


data['total_sqft'] = data['total_sqft'].apply(clean)


# In[52]:


data.total_sqft.unique()


# In[53]:


data.describe()


# In[54]:


data.dropna(inplace=True)


# In[55]:


data


# In[56]:


data['sqft_per_bed'] = data['total_sqft']/data['bedrooms']


# In[57]:


data


# In[58]:


data.sqft_per_bed.describe()


# In[59]:


data2 = data[data['sqft_per_bed'] >= 300]


# In[60]:


data2


# In[61]:


data2['price_per_sqft'] = data2['price']*100000/data2['total_sqft']


# In[62]:


data2


# In[63]:


data2['price_per_sqft'] = round(data2['price']*100000/data2['total_sqft'],2)


# In[64]:


data2


# In[65]:


data2.price_per_sqft.describe()


# In[66]:


data3 = data2[data2['price_per_sqft'] >= 2000]


# In[67]:


data3


# In[71]:


data3.drop(columns = ['size','sqft_per_bed','price_per_sqft'], axis = 1, inplace=True)


# In[72]:


data3


# In[73]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler


# In[74]:


from sklearn.model_selection import train_test_split


# In[75]:


from sklearn.linear_model import LinearRegression


# In[76]:


from sklearn.pipeline import make_pipeline


# In[78]:


from sklearn.compose import make_column_transformer


# In[79]:


col_trans = make_column_transformer((OneHotEncoder(sparse_output=False),['location']),remainder='passthrough')


# In[80]:


lr = LinearRegression()


# In[81]:


scaler = StandardScaler()


# In[82]:


model = make_pipeline(col_trans,scaler,lr)


# In[85]:


data_input = data3.drop(columns = ['price'])
data_output = data3['price']


# In[86]:


x_train,x_test,y_train,y_test = train_test_split(data_input, data_output, test_size=0.2)


# In[87]:


model.fit(x_train,y_train)


# In[90]:


model.score(x_test, y_test)


# In[103]:


input = pd.DataFrame([['Electronic City Phase II',10000.0,3.0,2.0,3]],columns = ['location','total_sqft','bath',	'balcony','bedrooms'])


# In[104]:


model.predict(input)


# In[105]:


import pickle as pk


# In[106]:


pk.dump(model,open('House_prediction_model.pkl','wb'))


# In[107]:


data3.to_csv('Cleaned_data.csv')


# In[ ]:




