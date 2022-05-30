import os
import time
import pandas as pd
from glove import Glove
from glove import Corpus
from skmultiflow.utils import calculate_object_size
from preprocessing.utils import read_log
from preprocessing.utils import retrieve_traces
from preprocessing.utils import average_feature_vector_glove


def prepare_traces(traces):
    for trace in traces:
        yield trace

def save_results(vector, dimension, ids, time, memory, y):
    out_df = pd.DataFrame(vector, columns=[f'feature_{i}' for i in range(dimension)])
    out_df['case'] = ids
    out_df['time'] = time
    out_df['memory'] = memory
    out_df['label'] = y

    return out_df

def glove_(datafile, df, dimension, word_aggregation):
    """
    Use the glove method
    
    Parameters
    ----------

    Returns
    ----------
    """

    #dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
   
    file_name = datafile.name.split('.xes')[0]    
    encoding_name = 'glove_'

    # read event log and import case id and labels
    ids, traces_, y = retrieve_traces(read_log(df))

    #for dimension in dimensions:
    start_time = time.time()
    # generate model
    model = Corpus()
    model.fit(prepare_traces(traces_))

    glove = Glove(no_components=dimension)
    glove.fit(model.matrix, epochs=10, no_threads=8)
    glove.add_dictionary(model.dictionary)
    
    # calculating the average feature vector for each sentence (trace)
    vectors_average, vectors_max = average_feature_vector_glove(glove, traces_)

    end_time = time.time() - start_time

    if word_aggregation == 'Average':
        mem_size = calculate_object_size(vectors_average) + calculate_object_size(model) + calculate_object_size(glove)
        out_df = save_results(vectors_average, dimension, ids, end_time, mem_size, y)

    else:    
        mem_size = calculate_object_size(vectors_max) + calculate_object_size(model) + calculate_object_size(glove)
        out_df = save_results(vectors_max, dimension, ids, end_time, mem_size, y)

    return out_df, file_name, encoding_name