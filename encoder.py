from category_encoders import OneHotEncoder, BaseNEncoder, TargetEncoder
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
    encoder = TargetEncoder(cols=columns)
    return Pipeline(steps=[('encoder', encoder)])
