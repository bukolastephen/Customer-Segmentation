import streamlit as st
import pandas as pd
import joblib

encoder= joblib.load('C:\Users\MINE\Desktop\Customer Segmentation\model\encoder.pkl')
scaler= joblib.load('C:\Users\MINE\Desktop\Customer Segmentation\model\scale.pkl')
model= joblib.load('C:\Users\MINE\Desktop\Customer Segmentation\model\model.pkl')

st.title('Customer Segmentation App')
st.write('Enter Customer Data')

df= pd.read_csv(r'C:\Users\MINE\Desktop\Customer Segmentation\data\clustered.csv')

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
  'amount_spent': amount_spent,
  'purchases': purchases,
  'NumWebVisitsMonth': web_visit,
  'campaign': campaign
}

df= pd.DataFrame(data, index=[1])

for col in df.select_dtypes(include='object').columns:
  df[col]= encoder.fit_transform(df[col])

df_scaled= scaler.transform(df)



button= st.button('Submit')

if button== True:
  cluster= model.predict(df_scaled)
  st.write(cluster)
  if cluster == 0:
    st.success('customer among those who has the lower avarage age in the distribution,where most are married with 2n Cycle education status.')
  elif cluster == 1:
    st.success('customer among those who make the most purchases and are also the second top spenders, most of the them are graduation holders and are married.')
  elif cluster == 2:
    st.success('customer among those who has the highest amount of Phd Holders and Married Folks, they also rank as the oldest.')
  elif cluster== 3:
    st.success('customer among those who visit the web the most, most are married and graduation holders.')
  elif cluster==4:
    st.success('customer among those who are the highest earners and spenders, they also receive the highest campaign.')
