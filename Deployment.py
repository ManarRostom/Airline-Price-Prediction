
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import category_encoders
import sklearn
import xgboost


Model = joblib.load('Final_Model.pkl')
Inputs_dict = joblib.load("Inputs_dict.pkl")

def Predict(Airline, Source, Destination, Route, Duration, Journey_Day, Journey_Month):
    df_test = pd.DataFrame(columns=Inputs_dict['inputs'])
    df_test.at[0,'Airline'] = Airline
    df_test.at[0,'Source'] = Source
    df_test.at[0,'Destination'] = Destination
    df_test.at[0,'Route'] = Route
    df_test.at[0,'Duration'] = Duration
    df_test.at[0,'Journey_Day'] = Journey_Day
    df_test.at[0,'Journey_MonthName'] = Journey_Month
    return Model.predict(df_test)[0]
    

def Main():
    st.markdown('<p style="font-size:50px;text-align:center;"><strong>Airline Ticket Price Prediction</strong></p>',unsafe_allow_html=True)
    col1_1, col1_2 = st.columns([2,2])
    col2_1, col2_2 = st.columns([2,2])    
    col3_1, col3_2 = st.columns([2,2])
    col4_1, col4_2 = st.columns([2,2])
    
    with col1_1:
        Airline = st.selectbox('Choose Airline Name : ',Inputs_dict['Airlines'])
    with col2_1:
        Source = st.selectbox('Choose Source of the Flight : ',Inputs_dict['Source'])
    with col3_1:  
        Destination = st.selectbox('Choose Destination of the Flight : ',Inputs_dict['Source_Destination'][Source])
    with col4_1:
        Route = st.selectbox('Choose Route of the Flight : ',Inputs_dict['Destination_Route'][Destination])
        
    with col1_2:
        Hours = st.slider('Duration of the flight in hours : ',max_value=24, min_value=1, step=1, value=2)
    with col2_2:
        Minutes = st.slider('Duration of the flight in Minutes : ',max_value=60, min_value=0, step=1, value=30)
    with col3_2:
        Journey_Day = st.slider('Journey Day : ',max_value=31, min_value=1, step=1, value=15)
    with col4_2:
        Journey_Month = st.slider('Journey Month : ',max_value=12, min_value=1, step=1, value=5)
    
    ## Calculate Duration 
    Duration = (Hours * 60) + Minutes
    
    if st.button('Predict'):
        res = Predict(Airline, Source, Destination, Route, Duration, Journey_Day, Journey_Month)
        ## Reverse Log 
        final_res = 10 ** res
        st.text(f'Predicted Price is : {int(final_res)}')
        
    
Main()
