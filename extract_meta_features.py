import os
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from pymfe.mfe import MFE
import warnings
warnings.filterwarnings('ignore')


def sort_alphanumeric(data):
    """
    Returns a alphanumeric sortered list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


fs = ['c1', 'c2', 'cls_coef', 'density', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 'lsc', 'n1', 'n2', 'n3', 't1', 't2', 't3', 't4']

metrics = []
encoding = 'graphwave'
embedding = 'node'
edge_type = '-'
aggregation = 'average'
size = 256
number = 71

path = f'{encoding}_{embedding}_{aggregation}/{size}'
print(f'{encoding} - {embedding} - {edge_type} - {aggregation} - {size}')

# time.sleep(10800)

for file in tqdm(sort_alphanumeric(os.listdir(path))):
    df = pd.read_csv(f'{path}/{file}')

    encoding_time = df['time'][0]
    encoding_memory = df['memory'][0]
    labels = list(df['label'])
    df.drop(['case', 'time', 'memory', 'label'], axis=1, inplace=True)
    # df['trace_is_fit'] = df['trace_is_fit'].astype(int)

### parte importante 
    # X = produced encoding, y = labels
    mfe = MFE(groups='complexity', features=fs)
    mfe.fit(X=df.to_numpy(), y=labels, transform_num=False)
    # ft = feature to analyze
    ft = mfe.extract()
### 

    scenario, log_size, anomaly_type, anomaly_percentage = file.split('.csv')[0].split('_')
    out_metrics = [file, scenario, log_size, anomaly_type, anomaly_percentage, embedding, edge_type, aggregation]
    out_metrics.extend([len(df.columns), encoding_time, encoding_memory])
    out_metrics.extend(ft[1])

    metrics.append(out_metrics)
    os.remove(f'{path}/{file}')

columns = ['log', 'scenario', 'log_size', 'anomaly_type', 'anomaly_percentage', 'embedding', 'edge_type', 'aggregation', 'feature_vector_size', 'encoding_time', 'encoding_memory']
columns.extend(ft[0])
df = pd.DataFrame(metrics, columns=columns)
df.to_csv(f'{encoding}_{number}.csv', index=False)
