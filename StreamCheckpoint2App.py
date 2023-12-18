# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:48:08 2023

@author: PC
"""

# import necessary libraries
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import LabelEncoder, minmax_scale
from PIL import Image

# load model data
model = load('Financial_Inclusion_in_Africa.joblib')
# Encode cat. variables
def cat_enc(df):
    le = LabelEncoder()
    df['country'] = le.fit_transform(df['country'])
    df['location_type'] = le.fit_transform(df['location_type'])
    df['cellphone_access'] = le.fit_transform(df['cellphone_access'])
    df['gender_of_respondent'] = le.fit_transform(df['gender_of_respondent'])
    df['relationship_with_head'] = le.fit_transform(df['relationship_with_head'])
    df['marital_status'] = le.fit_transform(df['marital_status'])
    df['education_level'] = le.fit_transform(df['education_level'])
    df['job_type'] = le.fit_transform(df['country'])
    return df
def norma_num(df):
    df['age_of_respondent'] = minmax_scale(df['age_of_respondent'])
    return df
def input_processor(inp_df):
    inp_df = cat_enc(inp_df)
    inp_df = norma_num(inp_df)
    return inp_df
def main():
    st.title('FINANCIAL INCLUSION PREDICTION APP')
    st.write('This App predicts the likelihood of an individual domiciled in some East-African countries, having or using a bank account depending on some environmental, social and individual variables.')
    img = Image.open('financial_incl.jpg')
    st.image(img, width=500)
    #store user input
    user_input = {}
    user_input['country'] = st.selectbox("Respondent's Country", ['Rwanda','Tanzania','Kenya','Uganda'])
    user_input['year'] = st.selectbox('Year', ['2016','2017','2018'])
    user_input['location_type'] = st.radio("select location type:", ('Rural','Urban'))
    user_input['cellphone_access'] = st.radio("Does respondent have access to cellphone?", ('True','False'))
    user_input['household_size'] = st.number_input("individual's household size", min_value=1,max_value=21, step=1)
    user_input['age_of_respondent'] = st.number_input("Respondent's Age", )
    user_input['gender_of_respondent'] = st.radio("Respondent's Gender", ('Male','Female'))
    user_input['relationship_with_head'] = st.selectbox("Respondent relationship with household head", ['Head of Household','Spouse','Child','Parent','Other relative'])
    user_input['marital_status'] = st.selectbox("Respondent's Marital Status", ['married/living together','single/never married','widowed','divorced/separated','dont know'])
    user_input['education_level'] = st.selectbox("Respondent's Educational Level", ['Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA'])
    user_input['job_type'] = st.selectbox("Respondent's Job Type", ['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])
    user_df = pd.DataFrame([user_input])
    st.write(user_df)
    if st.button('PREDICT'):
        final_df = input_processor(user_df)
        prediction = model.predict(final_df)[0]
        if prediction == 1:
            st.write('The Respondent is likely to have a Bank account')
        else:
            st.write("The Respondent likely DON'T have a bank account")
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
