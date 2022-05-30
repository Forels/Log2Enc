import os
import time
import pandas as pd
from tqdm import tqdm
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.log_skeleton import algorithm as lsk_discovery
from pm4py.algo.conformance.log_skeleton import algorithm as lsk_conformance
from skmultiflow.utils import calculate_object_size

def compute_alignments(alignments):
    no_dev_total, no_constr_total, dev_fitness, is_fit = [], [], [], []
    for alignment in alignments:
        no_dev_total.append(alignment['no_dev_total'])
        no_constr_total.append(alignment['no_constr_total'])
        dev_fitness.append(alignment['dev_fitness'])
        is_fit.append(alignment['is_fit'])

    return [no_dev_total, no_constr_total, dev_fitness, is_fit]

def log_skeleton(log, datafile, ids, traces, y, net, initial_marking, final_marking):
    """
    Use the log_skeleton method
    
    Parameters
    ----------

    Returns
    ----------
    """

    file_name = datafile.name.split('.xes')[0]
    encoding_name = "log_skeleton"
    
    # read event log and import case id and labels 
    
    # import log
    
    start_time = time.time()
    
    # generate process model
    
    # encoding
    skeleton = lsk_discovery.apply(log, parameters={lsk_discovery.Variants.CLASSIC.value.Parameters.NOISE_THRESHOLD: 0.05})
    conf_result = lsk_conformance.apply(log, skeleton)
    
    end_time = time.time() - start_time
    memory = calculate_object_size(conf_result) + calculate_object_size(skeleton)
    
    final_alignments = compute_alignments(conf_result)
    
    # saving
    out_df = pd.DataFrame()
    out_df['no_dev_total'] = final_alignments[0]
    out_df['no_constr_total'] = final_alignments[1]
    out_df['dev_fitness'] = final_alignments[2]
    out_df['is_fit'] = final_alignments[3]
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y

    return out_df, file_name, encoding_name
