import pickle

import pandas as pd


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function used to downcast values inside the dataframe
    :param df: pandas dataframe
    :return: lighter version of the same pandas dataframe
    """
    float_cols = [c for c in df if df[c].dtype in ["float64"]]
    int_cols = [c for c in df if df[c].dtype in ['int64']]
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int16')
    return df


def save_object(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def read_object(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
