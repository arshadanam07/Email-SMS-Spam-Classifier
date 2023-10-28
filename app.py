import streamlit as st
import pickle
import string
import nltk
import sklearn
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()



def transform_text(text):
    text = text.lower()  # this is for converting into lower
    text = nltk.word_tokenize(text)  # this is for convering into tokens
    y = []
    for i in text:  # this is for removing special characters
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # this is for removing stopwords and punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model_spam.pkl','rb'))

st.title("SPAM CLASSIFIER for email and sms")
input_sms= st.text_input("Enter the message")

if st.button('Predict'):



#1.pre-process
    transformed_sms = transform_text(input_sms)

#2.vectorize
    vector_input= tfidf.transform([transformed_sms])
#3.predict
    result= model.predict(vector_input)
#4.display

    if result==1:
        st.header("Spam:Delete it")
    else:
        st.header("Message:Reply!!")





