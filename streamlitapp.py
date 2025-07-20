import streamlit as st
import pandas as pd
import numpy as np

st.title("House Price Prediction")


import pandas as pd
import numpy as np

data = pd.read_csv('bengaluru_house_prices.csv')

data.drop(columns=['area_type','availability','society'],inplace=True)

data.dropna(inplace=True)

data.location.value_counts()

data['location'] = data['location'].apply(lambda x:x.strip())

data.location.value_counts()

location_stats = data.groupby('location')['location'].agg('count').sort_values(ascending=False)

location_less_than_10_entries = location_stats[location_stats <=10]

location_less_than_10_entries

data['location'] = data['location'].apply(lambda x:'other' if x in location_less_than_10_entries else x)

data['location'].value_counts()

data

data['size'].unique()

data['bedrooms'] = data['size'].apply(lambda x:int(x.split(' ')[0]))

data

data.total_sqft.unique()

def clean(sqft):
    tokens = sqft.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    else:
        try:
            return float(sqft)
        except:
            return None
            
    

data['total_sqft'] = data['total_sqft'].apply(clean)

data.total_sqft.unique()

data.describe()

data.dropna(inplace=True)

data

data['sqft_per_bed'] = data['total_sqft']/data['bedrooms']

data

data.sqft_per_bed.describe()

data2 = data[data['sqft_per_bed'] >= 300]

data2

data2['price_per_sqft'] = data2['price']*100000/data2['total_sqft']

data2

data2['price_per_sqft'] = round(data2['price']*100000/data2['total_sqft'],2)

data2

data2.price_per_sqft.describe()

data3 = data2[data2['price_per_sqft'] >= 2000]

data3

data3.drop(columns = ['size','sqft_per_bed','price_per_sqft'], axis = 1, inplace=True)

data3

from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer

col_trans = make_column_transformer((OneHotEncoder(sparse_output=False),['location']),remainder='passthrough')

lr = LinearRegression()

scaler = StandardScaler()

model = make_pipeline(col_trans,scaler,lr)

data_input = data3.drop(columns = ['price'])
data_output = data3['price']

x_train,x_test,y_train,y_test = train_test_split(data_input, data_output, test_size=0.2)

model.fit(x_train,y_train)

model.score(x_test, y_test)

input = pd.DataFrame([['Electronic City Phase II',10000.0,3.0,2.0,3]],columns = ['location','total_sqft','bath',	'balcony','bedrooms'])

model.predict(input)

import pickle as pk

pk.dump(model,open('House_prediction_model.pkl','wb'))

data3.to_csv('Cleaned_data.csv')

