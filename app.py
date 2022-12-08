import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps=PorterStemmer()


def func1(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    z=[]
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            z.append(i)
    l=[]
    for i in z:
        l.append(ps.stem(i))
    return ' '.join(l)

df=pickle.load(open('C:/Users/sande/OneDrive/Desktop/Python/bnb_model.pkl','rb'))
tfidf=pickle.load(open('C:/Users/sande/OneDrive/Desktop/Python/spam.pkl','rb'))
st.title('Email / Spam Detection')

input_sms=st.text_input('Enter the message')

if st.button('Predict'):

    #1 preprocess
    transform_sms=func1(input_sms)
    #2 vectorizer
    vector_in=tfidf.transform([transform_sms])
    #3 predict
    result=df.predict(vector_in)[0]
    #4 display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')