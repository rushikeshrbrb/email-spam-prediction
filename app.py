import streamlit as st
import pickle
import  string
import sklearn


import  nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):

  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y[:]   # cloanning of list y because of list is immutable we cannaot copied as it is
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

tfidf=pickle.load(open('vetorizer.pkl','rb'))
model=pickle.load(open('modl.pkl','rb'))
st.title("Email/SMS Spam Classifire")

input_sms=st.text_input("Enter the mssg")
if st.button("predict"):
  # st.button('predict')
  transformed_mssg=transform_text(input_sms)
  vector_input=tfidf.transform([transformed_mssg])
  result=model.predict(vector_input[0])
  if result==1:
      st.header("spam")
  else:
      st.header("not spam")

