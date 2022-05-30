import os
import time
import pandas as pd
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from skmultiflow.utils import calculate_object_size

def compute_alignments(replayed_traces):
    trace_is_fit, trace_fitness, missing_tokens, consumed_tokens, remaining_tokens, produced_tokens = [], [], [], [], [], []
    for replayed in replayed_traces:
        trace_is_fit.append(replayed['trace_is_fit'])
        trace_fitness.append(float(replayed['trace_fitness']))
        missing_tokens.append(float(replayed['missing_tokens']))
        consumed_tokens.append(float(replayed['consumed_tokens']))
        remaining_tokens.append(float(replayed['remaining_tokens']))
        produced_tokens.append(float(replayed['produced_tokens']))

    return [trace_is_fit, trace_fitness, missing_tokens, consumed_tokens, remaining_tokens, produced_tokens]


def tokenreplay(log, datafile, ids, traces, y, net, initial_marking, final_marking):
    """
    Use the token replay method
    
    Parameters
    ----------

    Returns
    ----------
    """

    file_name = datafile.name.split('.xes')[0]
    encoding_name = "tokenreplay"

    # read event log and import case id and labels
   
    # import xes log for process discovery

    start_time = time.time()

    # generate process model

    # calculating tokenreplay
    replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)

    end_time = time.time() - start_time
    memory = calculate_object_size(replayed_traces)

    final_token_replay = compute_alignments(replayed_traces)

    # saving
    out_df = pd.DataFrame()
    out_df['trace_is_fit'] = final_token_replay[0]
    out_df['trace_fitness'] = final_token_replay[1]
    out_df['missing_tokens'] = final_token_replay[2]
    out_df['consumed_tokens'] = final_token_replay[3]
    out_df['remaining_tokens'] = final_token_replay[4]
    out_df['produced_tokens'] = final_token_replay[5]
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y

    return out_df, file_name, encoding_name
