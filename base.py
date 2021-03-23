import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Telecommunications provider membership classification', page_icon="./f.png")
st.title('Telecommunications provider membership classification')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')
st.write("""
Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset,  with predefined labels, we need to build a model to be used to predict class of a new or unknown case. 

The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns. 

## About the dataset

The target field, called **custcat**, has four possible values that correspond to the four customer groups, as follows:
- 1- Basic Service
- 2- E-Service
- 3- Plus Service
- 4- Total Service

Our objective is to build a classifier, to predict the class of unknown cases. We will use a specific type of classification called K nearest neighbour.

""")

@st.cache
def load_data():
  return pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv")

df = load_data()
st.dataframe(df)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
y = df['custcat'].values

st.write("## Best K")
st.write("In the above chart, we can see that the model with most accuracy is when we set k=9")
from PIL import Image
im = Image.open("1.png")
st.image(im, "Accuracy according ks")

st.write("## Make your prediction!")
col1, col2 = st.beta_columns(2)

region = col1.selectbox("Region", [1,2,3])
marital = col1.selectbox("Is married?", ["No", "Yes"])
marital = 0 if marital == "No" else 1
tenure = col1.slider("Tenure", 1, 72, 23)
age = col1.slider("Age", 5, 90, 20)
address = col1.slider("Address", 0, 55, 10)

income = col2.number_input("Income",0, 2000, value=300)
ed = col2.slider("Education",0,4, value=2)
employ = col2.slider("Years of Employ", 0, 50, 23)
retire = col2.selectbox("Is retired?", ["No", "Yes"])
retire = 0 if retire == "No" else 1
gender = col2.selectbox("Sex", ["Male", "Female"])
gender = 0 if gender == "Male" else 1

reside = st.selectbox("Reside", range(1,9))

user_df = [[region, tenure, age, marital, address, income, ed, employ, retire, gender, reside]]
user_df = preprocessing.StandardScaler().fit_transform(pd.DataFrame(user_df).to_numpy()[0][:, np.newaxis]).reshape(1,-1)

if st.button("Predict"):
  model = pickle.load(open("KNN.pkl", "rb"))
  predictions = model.predict(user_df)
  predictions_proba = model.predict_proba(user_df)
  labels = ["Basic Service","E-Service", "Plus Service", "Total Service"]

  st.write("The given client belongs to the membership: ")
  col3, col4 = st.beta_columns(2)
  col3.write("Prediction")
  col3.dataframe(pd.DataFrame(pd.Series(labels[predictions[0]-1]), columns=["Value"]))
  col4.write("Probability")
  col4.dataframe(pd.DataFrame(predictions_proba, columns=labels))




# This app repository

st.write("""
## App repository

[Github](https://github.com/ftarantuviez/)TODO
""")
# / This app repository