import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Resume Classification System")

resume = st.text_area("Paste Resume Text")

if st.button("Predict Category"):
    result = model.predict([resume])[0]
    st.success(result)