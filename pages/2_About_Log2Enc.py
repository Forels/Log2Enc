from turtle import width
import streamlit as st
from PIL import Image


st.title("About Log2Enc")

st.write('''
Log2Enc is a web-based tool to compare encoding techniques applied to event logs. 
Traditionally, process mining techniques apply transformation steps to convert event log data to other formats, such as projecting traces in the feature space. 
However, depending on the application or data behavior, different encoding techniques could be applied to obtain optimal results. 
Log2Enc compares almost 30 encoding techniques from three families: process mining encodings, word embeddings and graph embeddings. 
To analyze and compare different methods, we apply several metrics to capture performance from complementary perspectives. 
The metrics measure data distribution, class overlap and separability, dimensionality, among others.\n
You can read more about it [here](https://air.unimi.it/handle/2434/821352).         
''')

st.subheader("Contact us")

icon = Image.open('./images/person-icon.png')
forello = Image.open('./images/forello.png')
tavares = Image.open('./images/tavares.png')
ceravolo = Image.open('./images/ceravolo.png')
col1, col2, col3 = st.columns(3)

with col1:
    st.image(icon, width=100)
    st.write("[prof. Paolo Ceravolo](https://sesar.di.unimi.it/staff/paolo-ceravolo/)")
with col2:
    st.image(icon, width=100)
    st.write("[dr. Gabriel Tavares](https://sesar.di.unimi.it/staff/gabriel-marques-tavares/)")
with col3:
    st.image(icon, width=100)
    st.write("Gionata Forello")
