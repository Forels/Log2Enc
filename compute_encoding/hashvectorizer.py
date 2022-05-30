import os
import time
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from skmultiflow.utils import calculate_object_size

def hashvectorizer(datafile, ids, traces, y, dimension):
    """
    Use the hashvectorizer method
    
    Parameters
    ----------

    Returns
    ----------
    """

    #dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
    file_name = datafile.name.split('.xes')[0]
    encoding_name = 'hashvectorizer'

    traces = [str (item) for item in traces]


    # read event log and import case id and labels
        

    #for dimension in dimensions:
    
    start_time = time.time()
    
    # generate model
    model = HashingVectorizer(n_features=dimension)
    encoding = model.fit_transform(traces)
    end_time = time.time() - start_time
    mem_size = calculate_object_size(encoding) + calculate_object_size(model)
    # saving
    out_df = pd.DataFrame(encoding.toarray(), columns=[f'feature_{i}' for i in range(encoding.toarray().shape[1])])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = mem_size
    out_df['label'] = y

    return out_df, file_name, encoding_name
        