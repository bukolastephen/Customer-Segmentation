import streamlit as st
import pandas as pd
import joblib

st.title('Customer Segmentation')

df= pd.read_csv(r'C:\Users\MINE\Desktop\Customer Segmentation\data\clustered.csv')
st.write(df.head(10))

age= st.number_input("Age")
Education= st.selectbox('Education',df['Education'].unique())
Marital_Status= st.selectbox('Marital_Status', df['Marital_Status'].unique())
Income= st.number_input('Income')
amount_spent= st.number_input('Amount Spent')
purchases= st.number_input('Number Of Purchases')
web_visit= st.number_input('Number Of Web Visits')
campaign= st.number_input('Number of accepted Campaign')

data= {
  'age':age,
  'Education': Education,
  'Marital_Status': Marital_Status,
  'Income': Income,
  'purchases': purchases,
  'NumWebvisitsMonth': web_visit,
  'campaign': campaign
}