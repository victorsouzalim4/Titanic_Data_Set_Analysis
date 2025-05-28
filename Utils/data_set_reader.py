import pandas as pd

def csv_reader(file_name):

    data = pd.read_csv(file_name)

    return data
