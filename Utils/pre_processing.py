import pandas as pd
from sklearn.preprocessing import LabelEncoder


def treat_data(
        data, 
        columns_to_remove = None, 
        essential_columns = None, 
        label_encoding_columns = None,
    ):

    if columns_to_remove:
        data = data.drop(columns_to_remove, axis=1, errors='ignore')
    
    if essential_columns:
        data = data.dropna(subset = essential_columns)

    if label_encoding_columns:
        for col in label_encoding_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    return data
