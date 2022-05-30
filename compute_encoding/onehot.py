import os
import time
from networkx.algorithms.chordal import _chordal_graph_cliques
import pandas as pd
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from skmultiflow.utils import calculate_object_size


def onehot(datafile, ids, traces, y):
    """
    Use the onehot method
    
    Parameters
    ----------

    Returns
    ----------
    """

    file_name = datafile.name.split('.xes')[0]
    encoding_name = 'onehot'

    traces = [str (item) for item in traces]

    # read event log and import case id and labels

    start_time = time.time()

    # onehot encode
    corpus = CountVectorizer(token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+').fit_transform(traces)
    onehot = Binarizer().fit_transform((corpus.toarray()))
    
    end_time = time.time() - start_time
    memory = calculate_object_size(onehot)

    # saving
    out_df = pd.DataFrame(onehot, columns=[f'feature_{i}' for i in range(onehot.shape[1])])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y

    return out_df, file_name, encoding_name
