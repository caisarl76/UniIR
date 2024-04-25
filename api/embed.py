import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os
import argparse
from omegaconf import OmegaConf
import json
import gc
import random

import numpy as np
from torch.cuda.amp import autocast
import transformers

import sys
sys.path.append('../src/')
sys.path.append('../src/common')
sys.path.append('./')
sys.path.append('/app/uniir/api/')

from utils import *

def embed(input_query, model_path='/data/model_weight.pth',  gpu=0):
    device = 'cpu' if gpu is None else 'cuda:%d' %gpu

    model, img_preprocess_fn, tokenizer = get_model(model_path, device)
    batch = preprocess_input(input_query, img_preprocess_fn, tokenizer, device)
    
    with torch.no_grad():
        embeddings = model.encode_multimodal_input(
            batch["txt_batched"], batch["image_batched"], batch["txt_mask_batched"], batch["image_mask_batched"]
        ).half().cpu().numpy()

    return embeddings


def main(args):
    device = 'cpu' if args.gpu is None else 'cuda:%d' %args.gpu

    model, img_preprocess_fn, tokenizer = get_model(args.model_path, device)
    batch = get_input(args.input_query, img_preprocess_fn, tokenizer, device)
    
    with torch.no_grad():
        embeddings = model.encode_multimodal_input(
            batch["txt_batched"], batch["image_batched"], batch["txt_mask_batched"], batch["image_mask_batched"]
        ).half().cpu().numpy()

    np.save(args.embed_path, embeddings)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Embeddings for MBEIR")
    parser.add_argument("--model_path", type=str, default='/data/model_weight.pth')
    parser.add_argument("--embed_path", type=str, default="/data/embed.npy")
    parser.add_argument('--input_query', type=str, default='/data/input_query.json')
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()

# COPY model and dataset
# COPY /DATA2/jhkim/models/uniir/CLIP_SF/Large/Instruct/ArxivQA12_train90000_val10000/clip_sf_epoch_19.pth /data/model_weight.pth
# COPY /DATA2/jhkim/mbeir_arxiv12_val_cand_pool.index /data/arxiv12_val.index
# COPY /DATA2/jhkim/multimodal/arxiv_qa/passage/mbeir_arxiv12_val_cand_pool.jsonl /data/arxiv12_val_pool.jsonl

if __name__ == "__main__":
    args = parse_arguments()
    main(args)