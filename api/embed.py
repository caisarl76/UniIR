import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os
import argparse
from omegaconf import OmegaConf
import json
import gc
import random
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.cuda.amp import autocast
import transformers

import sys
sys.path.append('../src/')
sys.path.append('../src/common')


import dist_utils
from dist_utils import ContiguousDistributedSampler
from utils import build_model_from_config
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    Mode,
)

from models.uniir_clip.clip_scorefusion.clip_sf import CLIPScoreFusion


def _load_and_preprocess_image(query_img_path, img_preprocess_fn, mbeir_data_dir='/data/multimodal/arxiv_qa/'):
    """Load an image given a path"""
    if not query_img_path:
        return None
    full_query_img_path = os.path.join(mbeir_data_dir, query_img_path)
    assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
    image = Image.open(full_query_img_path).convert("RGB")
    image = img_preprocess_fn(image)
    return image

padded_image = torch.zeros((3, 224, 224))  # Note: this is a black image
padded_txt = ""  # Note: this is an empty string


def _get_padded_text_with_mask(txt):
    return (txt, 1) if txt not in [None, ""] else (padded_txt, 0)

def _get_padded_image_with_mask(img):
    return (img, 1) if img is not None else (padded_image, 0)


def main(args):
    device = 'cpu' if args.gpu is None else 'cuda:%d'%args.gpu
    
    model_dir = args.model_path
    model = CLIPScoreFusion(model_name='ViT-L/14')
    model.load_state_dict(torch.load(model_dir)["model"])
    model.eval()

    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()
    model = model.to(device)
    
    with open(args.input_query, 'r') as f:
        query_input = json.load(f)
    
    query_img_path = query_input['image']
    query_txt = query_input['text']
    assert len(query_img_path) == len(query_txt)
    
    txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []
    for img, txt in zip(query_img_path, query_txt):
        padded_img, img_mask = _get_padded_image_with_mask(_load_and_preprocess_image(img, img_preprocess_fn=img_preprocess_fn))
        padded_txt, txt_mask = _get_padded_text_with_mask(txt)
        
        txt_list.append(padded_txt)
        txt_mask_list.append(txt_mask)
        img_list.append(padded_img)
        img_mask_list.append(img_mask)
        
    batch = {
        "txt_batched": tokenizer(txt_list),
        "image_batched": torch.stack(img_list, dim=0),
        "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
        "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
        }
    
    for k, v in batch.items():
        batch[k] = v.to(device)

    with torch.no_grad():
        embeddings = model.encode_multimodal_input(
            batch["txt_batched"], batch["image_batched"], batch["txt_mask_batched"], batch["image_mask_batched"]
        )
        
    embeddings = embeddings.half().cpu().numpy()
    np.save(args.embed_path, embeddings)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Embeddings for MBEIR")
    parser.add_argument("--model_path", type=str, default='/data/model_weight.pth')
    parser.add_argument("--embed_path", type=str, default="/data/embed.npy")
    parser.add_argument('--input_query', type=str, default='/data/input_query.json')
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()

# COPY model and dataset
# COPY /DATA2/jhkim/models/uniir/CLIP_SF/Large/Instruct/ArxivQA12_train90000_val10000/clip_sf_epoch_19.pth /data/model_weight.pth
# COPY /DATA2/jhkim/mbeir_arxiv12_val_cand_pool.index /data/arxiv12_val.index
# COPY /DATA2/jhkim/multimodal/arxiv_qa/passage/mbeir_arxiv12_val_cand_pool.jsonl /data/arxiv12_val_pool.jsonl

if __name__ == "__main__":
    args = parse_arguments()
    main(args)