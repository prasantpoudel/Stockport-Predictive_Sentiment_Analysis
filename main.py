import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import torch
import transformers
from transformers import AutoTokenizer, AutoModel

ps = PorterStemmer()
st.set_page_config(page_title='Twitter Sentimental Anlaysis',layout="wide")
st.subheader("Hi:wave:")
st.write("Small Project :panda_face:")
st.header('Twitter Sentimental Anlaysis base on Stock Price')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

model = pickle.load(open('model.pkl','rb'))
input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    transformed_sms=str(transformed_sms)
    
    result=model(transformed_sms)
    st.header(result)



st.write("Small Project project for Internship")
st.write("1.Prasant poudel")
st.write("2.Hemant Devkota") 
st.write("3.Khajjapa Biradar")         
           