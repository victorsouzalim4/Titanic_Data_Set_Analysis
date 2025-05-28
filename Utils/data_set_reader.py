import pandas as pd

def csv_reader(file_name, columns_to_remove = None, essential_columns = None):

    data = pd.read_csv(file_name)

    if columns_to_remove != None:
        data = data.drop(columns_to_remove, axis=1, errors='ignore')
    
    if essential_columns != None:
        data = data.dropna(subset = essential_columns)


    return data
