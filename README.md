# Richter's Predictor: Modeling Earthquake Damage
![Alt text](https://s3.amazonaws.com/drivendata-public-assets/nepal-quake-bm-2.JPG)
#### Can you predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal based on aspects of building location and construction? 
---
## Introduction
This is a model to predict the damage for buildings affected by the Nepal earthquake in 2025. The data is from the following competition: 
https://www.drivendata.org/competitions/57/nepal-earthquake/page/134/

## Requirements
Python version: `3.11.7`
```
pip install -r requirements.txt
```

## Usage
In the main.ipynb one can set different Parameters:
- `PARAMETER_TUNING`
- `MODEL_VALIDATION`
- `SAVE_MODEL`
- `DO_SUBMISSSION`


## Streamlit app
The streamlit app allows **making predictions for example buidlings**. Select the properties for a specific building and see its predicted damage grade.  

Run the streamlit app from the console with:

```
streamlit run prediction_app_st.py
``` 

