import streamlit as st
import pandas as pd
import joblib
import time


#  Loading model
scaler= joblib.load('model/scale.pkl')
model= joblib.load('model/model.pkl')



st.title('Customer Segmentation App')
st.subheader('Enter Customer Data')



# Collecting data from users
age= st.number_input("Age")
Education= st.selectbox('Education',['Basic','2n Cycle','Graduation','Master','PhD'])
Marital_Status= st.selectbox('Marital_Status', ['Married','Together','Divorced','Single','Widow','Alone','Absurd','YOLO'])
Income= st.number_input('Income')
amount_spent= st.number_input('Amount Spent')
purchases= st.number_input('Number Of Purchases')
web_visit= st.number_input('Number Of Web Visits')
campaign= st.number_input('Number of accepted Campaign')



#  Converting to a data frame
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




# Encoding Categorical Varribles
edu_dict={'PhD':1,'Master': 2,'Graduation': 3,'2n Cycle': 4,'Basic': 5}
df['Education']= df['Education'].map(edu_dict)

mstatus_dict= {'Married': 1,'Together':2, 'Divorced': 3,'Single': 4, 'Widow': 5, 'Alone':6, 'Absurd':7, 'YOLO':8 }
df['Marital_Status']= df['Marital_Status'].map(mstatus_dict)

#  Scaling
df_scaled= scaler.transform(df)

st.write(df_scaled)

# Prediction

with st.spinner("Analyzing customer..."):
  time.sleep(10)
  if True:
    cluster= model.predict(df_scaled)[0]
    if cluster == 0:
      st.success('customer among those who are the oldest in the group, they rank second lowest in Income,amount_spent and purchases,they have the highest number of phd and masters holders.')
    elif cluster == 1:
      st.success('customer among those that make the most Income and spends the most, they make the second highest purchases and they accepted campaign the most')
    elif cluster == 2:
      st.success('customer among those who rank second oldest and are mostly single,most are graduation holders.')
    elif cluster== 3:
      st.success('customer among those that are  youngest in age, lowest Income erners,spenders as well as purchases, most are married and graduation holders they visit the web the most while taking the lowest campaign')
    elif cluster==4:
      st.success('customer among those that make the most purchases and are married.')