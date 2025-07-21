import pandas as pd
import pickle as pk 
import streamlit as st 

model = pk.load(open('D:\python\House_price_prediction_accuracy (1).pkl.ipynb','rb'))

st.header('House Price Predictor')
data = pd.read_csv('D:\python\House_price_prediction_accuracy (1).csv.ipynb')
loc = st.selectbox('choose the location',data['location'].unique())

input = pd.DataFrame([['Electronic City Phase II',2440.0,3.0,2.0,4]],columns = ['location','total_sqft','bath',	'balcony','bedrooms'])

if st.button('Predict Price'):
    output = model.predict(input)
    out_str = 'Price of the House is' + str(output[0]*100000)
    
