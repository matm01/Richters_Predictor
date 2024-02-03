import streamlit as st
import pandas as pd
import numpy as np
import pickle
from src.dataloader import dataloader
import random

st.title("Earthquake Damage Prediction")

st.markdown("This app uses a model to predict the damage grade of buildings affected by the 2015 Gorkha earthquake in Nepal. Learn more about the data set at the [DrivenData website](https://www.drivendata.org/competitions/57/nepal-earthquake/).")

st.markdown("👉 **Enter the details of a building** and select 'Predict'.")

"---"

# Load the data
@st.cache_data
def load_data():
    X_train, y_train, X_test = dataloader()
    return X_train, y_train, X_test

# Load the pipeline
@st.cache_resource
def load_pipeline():
    xgb_pipeline = pickle.load(open('models/xgb_pipeline.pkl', 'rb'))
    return xgb_pipeline

X_train, y_train, X_test = load_data()


st.header('Building Details')

count_floors_pre_eq = st.radio('How many floors did the building have before the earthquake?',
                                        options = np.sort(X_train["count_floors_pre_eq"].unique()),
                                        horizontal = True)

col1, col2 = st.columns(2)

age_building = col1.slider('How old was the building?',
                                    min_value=X_train["age"].min(),
                                    max_value=200,
                                    value=20,
                                    format="%d years")


foundation_type = col2.selectbox('What was the foundation type?',
                                options = np.sort(X_train["foundation_type"].unique()))

col3, col4 = st.columns(2)

roof_type = col3.radio('What was the roof type?',
                                options = np.sort(X_train["roof_type"].unique()),
                                horizontal = True)

geo_level_1_id = col4.selectbox('Where was the house (geo level 1)?',
                                    options = np.sort(X_train["geo_level_1_id"].unique()))


if st.button('Predict'):
    xgb_pipeline = load_pipeline()
    input_data = pd.DataFrame({
        'count_floors_pre_eq': [count_floors_pre_eq],
        'age': [age_building],
        'roof_type': [roof_type],
        'geo_level_1_id': [geo_level_1_id],
        'foundation_type': [foundation_type]
    })

    # Fill remaining columns with random values
    for column in X_train.columns:
        if column not in input_data.columns:
            random_value = random.choice(X_train[column])
            input_data[column] = random_value

    prediction = xgb_pipeline.predict(input_data)

    damage_grade = prediction[0] + 1

    st.header('Damage Prediction')

    if damage_grade == 1:
        st.markdown("The building is predicted to have :green[**low damage (grade 1)**].")
    elif damage_grade == 2:
         st.markdown("The building is predicted to have :orange[**medium damage (grade 2)**].")
    elif damage_grade == 3:
        st.markdown("The building is predicted to have :red[**high damage (grade 3)**].")
