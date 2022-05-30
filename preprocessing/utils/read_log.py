import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

def read_log(df_raw: DataFrame, replace_space: str = '-'):
    """
    Reads event log and preprocess it

    Parameters
    -----------------------
    df_raw: DataFrame,
        dataframe
    replace_space: str,
        Replace space from activity name
    Returns
    -----------------------
    Processed event log containing the only the necessary columns for encoding
    """
    df_raw['case'] = df_raw['case:concept:name']

    df_raw['activity_name'] = df_raw['concept:name'].str.replace(' ', replace_space)
    
    #if 'label' not in df_raw:
    #    df_raw['label'] = np.nan
    #    df_raw['label'] = df_raw['label'].replace(np.nan, "normal")
    
    df_proc = df_raw[['case', 'activity_name', 'label']]
    return df_proc