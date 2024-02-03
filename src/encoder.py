
from category_encoders import OneHotEncoder, BaseNEncoder
from sklearn.preprocessing import TargetEncoder
import pandas as pd
from sklearn.pipeline import Pipeline

def get_onehot_encoder(columns: list):
    """
    Create a one-hot encoder for the specified columns and return it as part of a pipeline.
    :param columns: list of columns to be one-hot encoded
    :return: Pipeline containing the one-hot encoder
    """
    encoder = OneHotEncoder(cols=columns)
    return Pipeline(steps=[('encoder', encoder)])

def get_basen_encoder(columns: list):
    """
    Create a base-n encoder for the specified columns and return it as part of a pipeline.
    :param columns: list of columns to be base-n encoded
    :return: Pipeline containing the base-n encoder
    """
    encoder = BaseNEncoder(cols=columns)
    return Pipeline(steps=[('encoder', encoder)])


def get_target_encoder(columns: list):
    """
    Create a target encoder for the specified columns and return it as part of a pipeline.
    :param columns: list of columns to be target encoded
    :return: Pipeline containing the target encoder
    """
    encoder = TargetEncoder(target_type='multiclass')
    return Pipeline(steps=[('encoder', encoder)])
  

def encode_labels(data, reverse=False):
    """
    Encode the labels in the input data based on the 'damage_grade' column. 
    If reverse is False, encode the labels from 1, 2, 3 to 0, 1, 2 respectively. 
    If reverse is True, reverse the encoding by mapping 0 to 1, 1 to 2, and 2 to 3. Return the modified data.
    """
    if reverse == False:
        data['damage_grade'] = data.damage_grade.map({1: 0, 2: 1, 3: 2})
        return data

    # reverse the encoding
    else:
        return pd.DataFrame(data)[0].map({0: 1, 1: 2, 2: 3}) 

