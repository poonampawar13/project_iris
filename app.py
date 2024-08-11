# import the required libraries

import numpy as np
import pandas as pd
import pickle
import streamlit as st

# provide header name to browser
st.set_page_config(page_title = 'Iris Project Poonam' , layout =  'wide')

# Add a title in body of browser
st.title('Iris Project ')

# take sepal length, sepal width , petal length and petal width as input from user

sep_len = st.number_input('Sepal Length : ', min_value = 0.00, step=0.01)
sep_wid = st.number_input('Sepal Width : ', min_value = 0.00, step=0.01)
pet_len = st.number_input('Petal Length : ', min_value = 0.00, step=0.01)
pet_wid = st.number_input('petal Width : ', min_value = 0.00, step=0.01)

# Add a button for prediction

submit = st.button('Predict')

# add subheader for predictions 

st.subheader('Predictions Are : ')

# Create a function to predict species along  woth probability

def predict_species(scaler_path , model_path):
    with open(scaler_path,'rb') as file1:
        scaler=pickle.load(file1)
    with open(model_path , 'rb') as file2:
        model = pickle.load(file2)
    dct ={'SepalLengthCm':[sep_len],
          'SepalWidthCm':[sep_wid],
          'PetalLengthCm':[pet_len],
          'PetalWidthCm':[pet_wid]}
    xnew = pd.DataFrame(dct)
    xnew_pre = scaler.transform(xnew)
    pred=model.predict(xnew_pre)
    probs = model.predict_proba(xnew_pre)
    max_prob = np.max(probs)
    return pred , max_prob

# show the results in streamlit 

if submit:
    scaler_path = '/workspaces/project_iris/notebook.ipynb/scaler.pkl'
    model_path = '/workspaces/project_iris/notebook.ipynb/model.pkl'
    pred, max_prob = predict_species(scaler_path,model_path)
    st.subheader(f'Predicted Species is : {pred[0]}')
    st.subheader(f'Probability of Prediction : {max_prob :.4f}')
    st.progress(max_prob)