import typing_extensions
from pm4py.algo.conformance.alignments import algorithm as alignments
from pm4py.algo.conformance.alignments import variants
from skmultiflow.utils import calculate_object_size
import time
import os
import pandas as pd

def compute_alignments(alignments):
    cost, visited_states, queued_states, traversed_arcs, fitness = [], [], [], [], []
    for alignment in alignments:
        if alignment is None:
            cost.append(0)
            visited_states.append(0)
            queued_states.append(0)
            traversed_arcs.append(0)
            fitness.append(0)
        else:
            cost.append(alignment['cost'])
            visited_states.append(alignment['visited_states'])
            queued_states.append(alignment['queued_states'])
            traversed_arcs.append(alignment['traversed_arcs'])
            fitness.append(alignment['fitness'])

    return [cost, visited_states, queued_states, traversed_arcs, fitness]

parameters = {}
parameters[alignments.Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = 1


def alignment(log:object, datafile:object, ids, traces, y, net, initial_marking, final_marking):
    """
    Use the alignment method
    
    Parameters
    ----------

    Returns
    ----------
    """  
    file_name = datafile.name.split('.xes')[0]
    encoding_name = "alignment"

    start_time = time.time()

    # compute alignments
    trace_alignments = alignments.apply_log(log, net, initial_marking, final_marking, parameters=parameters, variant=variants.dijkstra_no_heuristics)

    end_time = time.time() - start_time
    memory = calculate_object_size(trace_alignments)

    final_alignments = compute_alignments(trace_alignments)

    # saving
    out_df = pd.DataFrame()
    out_df['cost'] = final_alignments[0]
    out_df['visited_states'] = final_alignments[1]
    out_df['queued_states'] = final_alignments[2]
    out_df['traversed_arcs'] = final_alignments[3]
    out_df['fitness'] = final_alignments[4]
    out_df['case'] = ids
    out_df['time'] = end_time
    out_df['memory'] = memory
    out_df['label'] = y

    return out_df, file_name, encoding_name
