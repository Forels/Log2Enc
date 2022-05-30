import os
import time
import pandas as pd
from gensim.models import Word2Vec
from skmultiflow.utils import calculate_object_size
from preprocessing.utils import read_log
from preprocessing.utils import retrieve_traces
from preprocessing.utils import train_text_model
from preprocessing.utils import average_feature_vector


def save_results(vector, dimension, ids, time, memory, y):
    out_df = pd.DataFrame(vector, columns=[f'feature_{i}' for i in range(dimension)])
    out_df['case'] = ids
    out_df['time'] = time
    out_df['memory'] = memory
    out_df['label'] = y

    return out_df

def word2vec_skipgram(datafile, df, dimension, word_aggregation):
    """
    Use the word2vec_skipgram method
    
    Parameters
    ----------

    Returns
    ----------
    """

    #dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
    
    file_name = datafile.name.split('.xes')[0]
    encoding_name = 'word2vec_skipgram'


    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(df))

    #for dimension in dimensions:
    start_time = time.time()
    # generate model
    model = Word2Vec(vector_size=dimension, window=3, min_count=1, sg=1, workers=-1)
    model = train_text_model(model, traces)

    # calculating the average feature vector for each sentence (trace)
    vectors_average, vectors_max = average_feature_vector(model, traces)

    end_time = time.time() - start_time

    if word_aggregation == 'Average':
        mem_size = calculate_object_size(vectors_average) + calculate_object_size(model)
        out_df = save_results(vectors_average, dimension, ids, end_time, mem_size, y)
    
    if word_aggregation == 'Max':
        mem_size = calculate_object_size(vectors_max) + calculate_object_size(model)
        out_df = save_results(vectors_max, dimension, ids, end_time, mem_size, y)

    return out_df, file_name, encoding_name  