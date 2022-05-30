import os
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from skmultiflow.utils import calculate_object_size



def countvectorizer(datafile, ids, traces, y):
    """
    Use the countvectorizer method
    
    Parameters
    ----------

    Returns
    ----------
    """

    file_name = datafile.name.split('.xes')[0]
    encoding_name = 'countvectorizer'
    
    traces = [str(item) for item in traces]

    # read event log and import case id and labels
    
    start_time = time.time()

    # generate model
    model = CountVectorizer(token_pattern='[^0-9]')
    encoding = model.fit_transform(traces)

    end_time = time.time() - start_time
    memory = calculate_object_size(encoding)

    # saving
    out_df = pd.DataFrame(encoding.toarray(), columns=[f'feature_{i}' for i in range(encoding.toarray().shape[1])])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y

    return out_df, file_name, encoding_name
