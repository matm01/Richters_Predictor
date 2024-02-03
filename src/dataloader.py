import pandas as pd
def dataloader():
    """
    Load the dataset from CSV files.

    Returns:
    X_train: pandas DataFrame
        Training dataset values
    y_train: pandas DataFrame
        Training dataset labels
    X_test: pandas DataFrame
        Test dataset values
    """
    # Load the dataset
    X_train = pd.read_csv('data/train_values.csv')
    y_train = pd.read_csv('data/train_labels.csv')
    X_test = pd.read_csv('data/test_values.csv')
    return X_train, y_train, X_test