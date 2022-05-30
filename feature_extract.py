import os
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from pymfe.mfe import MFE
import warnings
warnings.filterwarnings('ignore')


def feature_extract(df:object, file:str, encoding:str, metrics:list):
    """
    Extract the feature

    Parameters
    ----------
    df : object
        The dataframe
    file : str
        The file
    encoding : str
        The encoding methodf
    metrics : 
        The metrics
    Returns
    ----------
    Return the metrics and the features
    """    

    fs = ['c1', 'c2', 'cls_coef', 'density', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 'lsc', 'n1', 'n2', 'n3', 't1', 't2', 't3', 't4']

    encoding_time = df['time'][0]
    encoding_memory = df['memory'][0]
    labels = list(df['label'])
    df.drop(['case', 'time', 'memory', 'label'], axis=1, inplace=True)

    mfe = MFE(groups='complexity', features=fs)
    mfe.fit(X=df.to_numpy(), y=labels, transform_num=False)
    ft = mfe.extract()

    #scenario, log_size, anomaly_type, anomaly_percentage = file.split('.csv')[0].split('_')
    out_metrics = [file, encoding]
    out_metrics.extend([len(df.columns), encoding_time, encoding_memory])
    out_metrics.extend(ft[1])

    metrics.append(out_metrics)

    return metrics,ft

def feature_save(metrics, email:str, usage_num:int, ft:list, traces:int):
    """
    Save the feature in .csv file

    Parameters
    ----------
    metrics : 
        The metrics
    email : str
        The email insert by the user, it used to create the file name
    number : int
        The number of times the user used the tool, is used to create the file name  
    ft : list
        The list of the features
    traces : int

    Returns
    ----------
    """   
    columns = ['log', 'encoding', 'feature_vector_size', 'encoding_time', 'encoding_memory']
    columns.extend(ft[0])
    df = pd.DataFrame(metrics, columns=columns)

    df_length = len(df.index)
    
    # replace NaN value
    df['c1'] = df['c1'].replace(np.nan, 0)
    df['c2'] = df['c2'].replace(np.nan, 1)
    df['cls_coef']  = df['cls_coef'].replace(np.nan, 1)
    df['density']   = df['density'].replace(np.nan, 1)
    df['f1.mean']   = df['f1.mean'].replace(np.nan, 1)
    df['f1.sd']     = df['f1.sd'].replace(np.nan, 0)
    df['f1v.mean']  = df['f1.mean'].replace(np.nan, 1)
    df['f1v.sd']    = df['f1v.sd'].replace(np.nan, 0)
    df['f2.mean']   = df['f2.mean'].replace(np.nan, 1)
    df['f2.sd']     = df['f2.sd'].replace(np.nan, 0)
    df['f3.mean']   = df['f3.mean'].replace(np.nan, 1)
    df['f3.sd']     = df['f3.sd'].replace(np.nan, 0)
    df['f4.mean']   = df['f4.mean'].replace(np.nan, 1)
    df['f4.sd']     = df['f4.sd'].replace(np.nan, 0)
    df['l1.mean']   = df['l1.mean'].replace(np.nan, 1)
    df['l1.sd']     = df['l1.sd'].replace(np.nan, 0)
    df['l2.mean']   = df['l2.mean'].replace(np.nan, 1)
    df['l2.sd']     = df['l2.sd'].replace(np.nan, 0)
    df['l3.mean']   = df['l3.mean'].replace(np.nan, 1)
    df['l3.sd']     = df['l3.sd'].replace(np.nan, 0)
    df['lsc']       = df['lsc'].replace(np.nan, len(traces))
    df['n1']        = df['n1'].replace(np.nan, 1)
    df['n2.mean']   = df['n2.mean'].replace(np.nan, 1)
    df['n2.sd']     = df['n2.sd'].replace(np.nan, 0)
    df['n3.mean']   = df['n3.mean'].replace(np.nan, 1)  
    df['n3.sd']     = df['n3.sd'].replace(np.nan, 0)
    df['t1.mean']   = df['t1.mean'].replace(np.nan, 1)
    df['t1.sd']     = df['t1.sd'].replace(np.nan, 0)
    
    for i in range(df_length):
        if df.loc[i, 't2'] == np.nan:
            df.loc[i, 't2'] = df.loc[i, 'feature_vector_size']
        if df.loc[i, 't3'] == np.nan:
            df.loc[i, 't3'] = df.loc[i, 'feature_vector_size']
    
    df['t4'] = df['t4'].replace(np.nan, 1)
    
    if not os.path.exists("meta_features_extracted"):
        os.makedirs("meta_features_extracted")
    
    # delete file if already exist 
    if os.path.exists(f'./meta_features_extracted/{email}_{usage_num}.csv'):
        os.remove(f'./meta_features_extracted/{email}_{usage_num}.csv')
    
    df.to_csv(f'./meta_features_extracted/{email}_{usage_num}.csv', index=False)   

    
