from preprocessing.utils import read_log
from preprocessing.utils import retrieve_traces
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

def process_model(df, log):
    
    # read event log and import case id and labels
    ids, traces, y = retrieve_traces(read_log(df))  

    # import log (already done)
    # generate process model
    net, initial_marking, final_marking = inductive_miner.apply(log)

    # encoding (alignment, tokenreplay, log_skeleton)

    # saving 

    return ids, traces, y, net, initial_marking, final_marking