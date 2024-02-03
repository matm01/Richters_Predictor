import pandas as pd
def clean_data(X_train, y_train):
    """
    Clean the input data by merging X_train and y_train on 'building_id' and removing outlier rows where 'age' has values > 900. 
    Then split X_train and y_train again and return the cleaned X_train and y_train.
    Parameters:
    - X_train: the input training data
    - y_train: the target training data
    Return:
    - X_train: the cleaned input training data
    - y_train: the cleaned target training data
    """

    merged_df = pd.merge(X_train, y_train, on='building_id', how='left')

    # Remove outliers rows where 'age' has values > 900
    merged_df = merged_df[merged_df['age'] <= 900]

    # Splitting X_train and y_train again
    X_train = merged_df.drop(columns=['damage_grade'])
    y_train = merged_df[['building_id', 'damage_grade']]
    return X_train, y_train