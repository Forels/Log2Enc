from preprocessing.utils import read_log
from preprocessing.utils import retrieve_traces
from preprocessing.utils import convert_traces_mapping
from preprocessing.utils import create_graph
import numpy as np
import networkx as nx


def graph_embeddings(datafile, df, log):

    # create graph
    graph = create_graph(log)
    mapping = dict(zip(graph.nodes(), [i for i in range(len(graph.nodes()))]))
    graph = nx.relabel_nodes(graph, mapping)

    # read event log, import case id and labels and transform activities names
    ids, traces_raw, y = retrieve_traces(read_log(df, ' '))
    traces = convert_traces_mapping(traces_raw, mapping)

    #print("graph nodes: ", len(graph.nodes())) 
    
    # generate model

    # calculating the average and max feature vector for each trace

    # saving

    return graph, ids, traces, y
