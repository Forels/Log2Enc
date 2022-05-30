import os
import time
import pandas as pd
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from skmultiflow.utils import calculate_object_size
from preprocessing.utils import retrieve_traces
from preprocessing.utils import read_log
from preprocessing.utils import average_feature_vector_doc2vec
from preprocessing.utils import sort_alphanumeric

def doc2vec(datafile, df, dimension):
    """
    Use the doc2vec method
    
    Parameters
    ----------

    Returns
    ----------
    """

    #dimensions = [2, 4, 8, 16, 32, 64, 128, 256]

    file_name = datafile.name.split('.xes')[0]
    encoding_name = 'doc2vec'

    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(df,' '))

    tagged_traces = [TaggedDocument(words=act, tags=[str(i)]) for i, act in enumerate(traces)]

    #for dimension in dimensions:
                
    start_time = time.time()
    
    # generate model
    model = Doc2Vec(vector_size=dimension, min_count=1, window=3, dm=1, workers=-1)
    model.build_vocab(tagged_traces)
    vectors = average_feature_vector_doc2vec(model, traces)
    
    end_time = time.time() - start_time
    mem_size = calculate_object_size(vectors) + calculate_object_size(model)
    
    # saving
    out_df = pd.DataFrame(vectors, columns=[f'feature_{i}' for i in range(dimension)])
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = mem_size
    out_df['label'] = y

    return out_df, file_name, encoding_name
