import streamlit as st
import pandas as pd
import joblib
import time

scaler= joblib.load('model/scale.pkl')
model= joblib.load('model/model.pkl')

st.title('Customer Segmentation App')
st.subheader('Enter Customer Data')

df= pd.read_csv('data/clustered.csv')

age= st.number_input("Age")
Education= st.selectbox('Education',df['Education'].unique())
Marital_Status= st.selectbox('Marital_Status', df['Marital_Status'].unique())
Income= st.number_input('Income')
amount_spent= st.number_input('Amount Spent')
purchases= st.number_input('Number Of Purchases')
web_visit= st.number_input('Number Of Web Visits')
campaign= st.number_input('Number of accepted Campaign')


df= pd.DataFrame({
  'age':age,
  'Education': Education,
  'Marital_Status': Marital_Status,
  'Income': Income,
  'amount_spent': amount_spent,
  'purchases': purchases,
  'NumWebVisitsMonth': web_visit,
  'campaign': campaign},
  index=[1])





edu_dict={'PhD':1,'Master': 2,'Graduation': 3,'2n Cycle': 4,'Basic': 5}
df['Education']= df['Education'].map(edu_dict)

mstatus_dict= {'Married': 1,'Together':2, 'Divorced': 3,'Single': 4, 'Widow': 5, 'Alone':6, 'Absurd':7, 'YOLO':8 }
df['Marital_Status']= df['Marital_Status'].map(mstatus_dict)

st.write(df)

df_scaled= scaler.transform(df)



button= st.button('Submit')
with st.spinner("Analyzing customer..."):
  time.sleep(10)
  if button:
    cluster= model.predict(df_scaled)[0]
    st.write(cluster)
    if cluster == 0:
      st.success('customer among those who are the second highest earners and spenders,they also rank third in number of purchases.')
    elif cluster == 1:
      st.success('customer among those who they rank oldest with the second least income, amount spent and number of purchases, also the highest number of Phd and masters holders.')
    elif cluster == 2:
      st.success('customer among those who are the  highest earners, campaings accepted and spenders,they also rank second in number of purchases.')
    elif cluster== 3:
      st.success('customer among those with the highest number of purchases and  second hisgest number of web visits.')
    elif cluster==4:
      st.success('customer among those who they have highest number of web visits, the youngest in age and earns the lowest, with the highest number of graduation, basic and 2n cycle holders and are mostly married')
