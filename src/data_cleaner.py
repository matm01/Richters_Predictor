import pandas as pd
def clean_data(X_train, y_train):

    merged_df = pd.merge(X_train, y_train, on='building_id', how='left')

    # Remove outliers rows where 'age' has values > 900
    merged_df = merged_df[merged_df['age'] <= 900]

    # Splitting X_train and y_train again
    X_train = merged_df.drop(columns=['damage_grade'])
    y_train = merged_df[['building_id', 'damage_grade']]
    return X_train, y_train