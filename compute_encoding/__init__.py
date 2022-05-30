from compute_encoding.deepwalk import deepwalk
from .alignment import alignment
from .tokenreplay import tokenreplay
from .log_skeleton import log_skeleton

from .countvectorizer import countvectorizer
from .doc2vec import doc2vec
from .hashvectorizer import hashvectorizer
#from .glove_ import glove_
from .onehot import onehot
from .tfidfvectorizer import tfidfvectorizer
from .word2vec_cbow import word2vec_cbow
from .word2vec_skipgram import word2vec_skipgram

from .boostne import boostne
from .deepwalk import deepwalk
from .diff2vec import diff2vec
from .glee import glee
from .graphwave import graphwave
from .grarep import grarep
from .hope import hope
from .laplacianeigenmaps import laplacianeigenmaps
from .netmf import netmf
from .nmfadmm import nmfadmm
from .node2vec_ import node2vec_
from .nodesketch import nodesketch 
from .role2vec import role2vec
from .walklets import walklets

__all__ = [
    "alignment",
    "tokenreplay",
    "log_skeleton",

    "countvectorizer",
    "doc2vec",
    "hashvectorizer",
    #"glove_",
    "onehot",
    "tfidfvectorizer",
    "word2vec_cbow",
    "word2vec_skipgram",
    
    "boostne",
    "deepwalk",
    "diff2vec",
    "glee",
    "graphwave",
    "grarep",
    "hope",
    "laplacianeigenmaps",
    "netmf",
    "nmfadmm",
    "node2vec_",
    "nodesketch",
    "role2vec",
    "walklets"
]