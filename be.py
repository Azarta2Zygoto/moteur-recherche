import json
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import warnings

warnings.filterwarnings("ignore")

# For embeddings and similarity computation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print("Required libraries imported successfully!")
except ImportError as e:
    print(f"Missing library: {e}")
    print(
        "Please install with: pip install sentence-transformers scikit-learn networkx"
    )

np.random.seed(42)


def load_corpus(file_path: str) -> dict[str, dict]:
    """
    TODO

    Load corpus data from JSONL file.
    Returns dictionary mapping document IDs to document data.
    """
    corpus = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            obj = json.loads(line)
            docid = str(obj["_id"])
            corpus[docid] = obj
    return corpus


def load_queries(file_path: str) -> dict[str, dict]:
    """
    TODO

    Load query data from JSONL file.
    Returns dictionary mapping query IDs to query data.
    """
    queries = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            obj = json.loads(line)
            qid = str(obj["_id"])
            queries[qid] = obj
    return queries


def load_qrels(file_path: str) -> dict[str, dict[str, int]]:
    """
    TODO

    Load relevance judgments from TSV file.
    Returns dictionary mapping query IDs to candidate relevance scores.
    """
    qrels = defaultdict(dict)
    with open(file_path, "r") as f:
        for line in f:
            qid, docid, score = line.strip().split("\t")
            qrels[qid][docid] = int(score)
    return qrels
