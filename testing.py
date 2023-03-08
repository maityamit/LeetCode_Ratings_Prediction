import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns


df = pd.read_csv(r'C:\Users\AMIT\Downloads\leetcode_contest_dataset.csv')

# HEADINGS
st.title('LeetCode Rank Predictor')
st.sidebar.header('User Given Data')
# st.subheader('Training Data Stats')
# st.write(df.describe())


# X AND Y DATA
x = df.drop(['final'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1, random_state = 0)

# FUNCTION
def user_report():
  initial = st.sidebar.slider('Initial Rating', 1000,3000, 1000 )
  rank = st.sidebar.slider('Your Rank', 0,30000, 0 )
  total = st.sidebar.slider('Total Participants', 0,30000, 0 )
  question = st.sidebar.slider('Question Attempt', 0,4, 0 )

  user_report_data = {
      'initial':initial,
      'rank':rank,
      'total':total,
      'question':question
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# User DATA
user_data = user_report()
st.subheader('User Data')
st.write(user_data)




# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)




# OUTPUT
st.subheader('Your New Ratings Will be: ')
output = user_result[0]
st.title(output)
# st.subheader('Accuracy: ')
# st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')
