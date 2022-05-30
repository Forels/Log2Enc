from preprocessing.utils import read_log
from preprocessing.utils import extract_corpus


def word_embeddings(df):

    # read event log and import case id and labels
    ids, traces, y = extract_corpus(read_log(df,' '))
  
    # generate model

    # saving

    traces = [str (item) for item in traces]

    return ids, traces, y