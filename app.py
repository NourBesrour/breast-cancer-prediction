import streamlit as st
import numpy as np
import pandas as pd
import pickle
from main import LR

# Charger le modèle
with open("xgbpipe.joblib", "rb") as f:
    model = pickle.load(f)

st.title("Benign or Malignant ? ")
st.image("BC.png", caption='BreastCancer')

st.markdown(" Whether the patient is pro or postmenopausal at the time diagnose,0 means that the patient has reached meanopause and 1 means that the patient has not reached menopause yet. ")
menopause = st.selectbox("Menopause", (0, 1))

st.markdown(" The number of axillary lymph nodes that contain metastatic, 1 means present and 0 means absent")
IN = st.selectbox("involved nodes", (0, 1))

st.markdown(" If it occurs on the left or right side, 1 means the cancer has spred and 0 means it hasn't spread yet")
breast = st.selectbox("breast", (0, 1))

st.markdown("If the cancer has spread to other part of the body or organ.")
metastatic = st.selectbox("metastatic", (0, 1))

st.markdown(" The gland is divided into 4 sections with nipple as a central point")
BQ = st.selectbox("breast quadrant", (0, 1))

st.markdown("If the patient has any history or family history on cancer, 1 means there is a history of cancer , 0 means no history")
History = st.selectbox("History", (0, 1))

st.markdown("age : age of the patient at the time of diagnose ")
age = st.number_input("Enter a positive integer", value=0, step=1)
if age < 0 or age % 1 != 0:
    st.warning("Please enter a positive integer.")
else:
    st.success("You entered a positive integer.")

st.markdown("Put the size of the tumor")
TS = st.number_input("tumor size")

columns = ['Age','Menopause','TS','Inv-Nodes','Breast','Metastasis','BQ','History']

def predict():
    row = np.array([menopause, IN, breast, metastatic, BQ, History, age, TS])
    X = pd.DataFrame([row], columns=columns)
    prediction = LR.predict(X)[0]
    if prediction == 1:
        st.success("Malignant")
    else:
        st.error("Benign")

st.button('Predict',on_click=predict)