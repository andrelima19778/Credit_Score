from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_10082024')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    #predictions = predictions_df['Label'][0]
    predictions = predictions_df['prediction_label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.png')
    image_credit = Image.open('bigstock-Credit-Score-Concept-Business.jpg')

    st.image(image,use_column_width=False)

    #add_selectbox = st.sidebar.selectbox(
    #"How would you like to predict?",
    #("Online", "Em lote"))

    st.sidebar.info('This app is created to predict credit score for customer')
    st.sidebar.success('Módulo 38 Exercício 2')
    
    st.sidebar.image(image_credit)

    st.title("Credit Score Prediction App")

    st.subheader("Modelo do PyCaret")

    st.markdown("**Modelo de Classificação produzido no pacote PyCaret**")

    st.write("Este app usa as 3 principais variáveis para predizer se um cliente é um "\
             "bom ou mau cliente, utilizando 50 mil linhas aleatórias da base de dados fornecida. O modelo não necessita ser fitado.")
    
    file_upload = st.file_uploader("Upload ftr file for predictions", type=["ftr"])

    if file_upload is not None:
        data = pd.read_feather(file_upload)
        data = data.sample(50000)
        predictions = predict_model(estimator=model,data=data)
        st.write(predictions)

if __name__ == '__main__':
    run()