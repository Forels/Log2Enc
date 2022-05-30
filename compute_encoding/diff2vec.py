import os
import time
import pandas as pd
from multiprocessing import cpu_count
from skmultiflow.utils import calculate_object_size
from karateclub.node_embedding.neighbourhood import Diff2Vec
from preprocessing.utils import trace_feature_vector_from_nodes
from preprocessing.utils import trace_feature_vector_from_edges
import warnings
warnings.filterwarnings('ignore')

from time import sleep
from tqdm import tqdm


def save_results(vector, dimension, ids, time, memory, y):
    out_df = pd.DataFrame(vector, columns=[f'feature_{i}' for i in range(dimension)])
    out_df['case'] = ids
    out_df['time'] = time
    out_df['memory'] = memory
    out_df['label'] = y
    
    return out_df

def diff2vec(datafile, graph, ids, traces, y, dimension, graph_aggregation):
    """
    Use the diff2vec method
    
    Parameters
    ----------

    Returns
    ----------
    """

    n_workers = cpu_count()

    #dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
    
    file_name = datafile.name.split('.xes')[0]     
    encoding_name = 'diff2vec'    

    # create graph

    # read event log, import case id and labels and transform activities names

    start_time = time.time()
    
    # generate model
    # model = Diff2Vec(dimensions=dimension)
    model = Diff2Vec(diffusion_cover=graph.number_of_nodes(), dimensions=dimension, workers=n_workers)
    model.fit(graph)
    training_time = time.time() - start_time
    
    # calculating the average and max feature vector for each trace
    start_time = time.time()
    node_average, node_max = trace_feature_vector_from_nodes(model.get_embedding(), traces, dimension)
    node_time = training_time + (time.time() - start_time)
    
    # saving
    if graph_aggregation == 'Node/Average':
        mem_size = calculate_object_size(node_average) + calculate_object_size(model)
        out_df = save_results(node_average, dimension, ids, node_time, mem_size, y)
    if graph_aggregation == 'Node/Max':
        mem_size = calculate_object_size(node_max) + calculate_object_size(model)
        out_df = save_results(node_max, dimension, ids, node_time, mem_size, y)
    
    start_time = time.time()
    edge_average_average, edge_average_max, edge_hadamard_average, edge_hadamard_max, edge_weightedl1_average, edge_weightedl1_max, edge_weightedl2_average, edge_weightedl2_max = trace_feature_vector_from_edges(model.get_embedding(), traces, dimension)
    edge_time = training_time + (time.time() - start_time)
    
    if graph_aggregation == "Edge/Average/Average":
        mem_size = calculate_object_size(edge_average_average) + calculate_object_size(model)
        out_df = save_results(edge_average_average, dimension, ids, edge_time, mem_size, y)
    
    if graph_aggregation == 'Edge/Average/Max':
        mem_size = calculate_object_size(edge_average_max) + calculate_object_size(model)
        out_df = save_results(edge_average_max, dimension, ids, edge_time, mem_size, y)
    
    if graph_aggregation == 'Edge/Hadamard/Average':
        mem_size = calculate_object_size(edge_hadamard_average) + calculate_object_size(model)
        out_df = save_results(edge_hadamard_average, dimension, ids, edge_time, mem_size, y)
    if graph_aggregation == 'Edge/Hadamard/Max':
        mem_size = calculate_object_size(edge_hadamard_max) + calculate_object_size(model)
        out_df = save_results(edge_hadamard_max, dimension, ids, edge_time, mem_size, y)
    
    if graph_aggregation == 'Edge/Wightedl1/Average':
        mem_size = calculate_object_size(edge_weightedl1_average) + calculate_object_size(model)
        out_df = save_results(edge_weightedl1_average, dimension, ids, edge_time, mem_size, y)
    if graph_aggregation == 'Edge/Wightedl1/Max':
        mem_size = calculate_object_size(edge_weightedl1_max) + calculate_object_size(model)
        out_df = save_results(edge_weightedl1_max, dimension, ids, edge_time, mem_size, y)
    
    if graph_aggregation == 'Edge/Wightedl2/Average':
        mem_size = calculate_object_size(edge_weightedl2_average) + calculate_object_size(model)
        out_df = save_results(edge_weightedl2_average, dimension, ids, edge_time, mem_size, y)
    if graph_aggregation == 'Edge/Wightedl2/Max':
        mem_size = calculate_object_size(edge_weightedl2_max) + calculate_object_size(model)
        out_df = save_results(edge_weightedl2_max, dimension, ids, edge_time, mem_size, y)

    return out_df, file_name, encoding_name
