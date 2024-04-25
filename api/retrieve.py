import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
from json import loads
import numpy as np

import faiss
import sys
sys.path.append('../src/')
sys.path.append('../src/common')
sys.path.append('./')
sys.path.append('/app/uniir/api/')

from utils import *

def retrieve(query_embedding, index_path="/data/arxiv12_val.index", pool_path="/data/arxiv12_val_pool.jsonl", num_cand_to_retrieve=5):
    query_embedding = query_embedding.astype('float32')
    ## 2. retrieve query_index 
    faiss.normalize_L2(query_embedding)
    index_cpu = faiss.read_index(index_path)
    ngpus = faiss.get_num_gpus()
    print(f"Faiss: Number of GPUs used for searching: {ngpus}")
    
    # co = faiss.GpuMultipleClonerOptions()
    # co.shard = True  # Use shard to divide the data across the GPUs
    # index_gpu = faiss.index_cpu_to_all_gpus(index_cpu, co=co, ngpu=ngpus)  # This shards the index across all GPUs
    index_gpu = index_cpu
    dist, indices = search_index_with_batch(query_embedding, index_gpu, num_cand_to_retrieve=num_cand_to_retrieve)
    n_query = indices.shape[0]
    print('Number of Queries: %d' %n_query)
    
    pool_dict = {}
    with open(pool_path, "r") as f2r:
        for each_line in f2r:
            pool_dict[loads(each_line)['did'].replace(':','00')] = loads(each_line) 

    retrieval_results = []
    for i in range(n_query):
        retrieval_results.append([pool_dict[str(id)] for id in list(indices[i])])
    
    
    return retrieval_results, None

def main(args):
    ## 1. Make query into index
    ### (1) Load query embeding
    query_embedding = np.load(args.embed_path).astype('float32')

    ## 2. retrieve query_index 
    faiss.normalize_L2(query_embedding)
    index_cpu = faiss.read_index(args.index_path)
    ngpus = faiss.get_num_gpus()
    print(f"Faiss: Number of GPUs used for searching: {ngpus}")
    
    # co = faiss.GpuMultipleClonerOptions()
    # co.shard = True  # Use shard to divide the data across the GPUs
    # index_gpu = faiss.index_cpu_to_all_gpus(index_cpu, co=co, ngpu=ngpus)  # This shards the index across all GPUs
    index_gpu = index_cpu
    dist, indices = search_index_with_batch(query_embedding, index_gpu, num_cand_to_retrieve=args.num_cand_to_retrieve)
    n_query = indices.shape[0]
    print('Number of Queries: %d' %n_query)
    
    pool_dict = {}
    with open(args.pool_path, "r") as f2r:
        for each_line in f2r:
            pool_dict[loads(each_line)['did'].replace(':','00')] = loads(each_line) 

    retrieval_results = []
    for i in range(n_query):
        retrieval_results.append([pool_dict[str(id)] for id in list(indices[i])])
    
    return retrieval_results

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Embeddings for MBEIR")
    parser.add_argument("--embed_path", type=str, default="/data/embed.npy")
    parser.add_argument("--index_path", type=str, default="/data/arxiv12_val.index")
    parser.add_argument("--pool_path", type=str, default="/data/arxiv12_val_pool.jsonl")
    parser.add_argument("--num_cand_to_retrieve", type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)