import os
import json
import torch
import numpy as np
from PIL import Image

import sys
sys.path.append('/app/uniir/src/')
from models.uniir_clip.clip_scorefusion.clip_sf import CLIPScoreFusion

padded_image = torch.zeros((3, 224, 224))  # Note: this is a black image
padded_txt = ""  # Note: this is an empty string

def _load_and_preprocess_image(query_img_path, img_preprocess_fn, mbeir_data_dir='/data/multimodal/arxiv_qa/'):
    """Load an image given a path"""
    if not query_img_path:
        return None
    full_query_img_path = os.path.join(mbeir_data_dir, query_img_path)
    assert os.path.exists(full_query_img_path), f"Image Path {full_query_img_path} does not exist"
    image = Image.open(full_query_img_path).convert("RGB")
    image = img_preprocess_fn(image)
    return image


def _get_padded_text_with_mask(txt):
    return (txt, 1) if txt not in [None, ""] else (padded_txt, 0)

def _get_padded_image_with_mask(img):
    return (img, 1) if img is not None else (padded_image, 0)


def get_model(model_path, device):
    model = CLIPScoreFusion(model_name='ViT-L/14')
    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()
    
    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()
    model = model.to(device)
    
    return model, img_preprocess_fn, tokenizer

def preprocess_input(query_input, img_preprocess_fn, tokenizer, device='cpu'):
    query_img_path = query_input['image']
    query_txt = query_input['text']
    
    txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []
    padded_img, img_mask = _get_padded_image_with_mask(_load_and_preprocess_image(query_img_path, img_preprocess_fn=img_preprocess_fn))
    padded_txt, txt_mask = _get_padded_text_with_mask(query_txt)
    
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
    return batch

def get_input(input_path, img_preprocess_fn, tokenizer, device='cpu'):
    with open(input_path, 'r') as f:
        query_input = json.load(f)
    return preprocess_input(query_input, img_preprocess_fn, tokenizer, device='cpu')

def search_index_with_batch(query_embeddings_batch, index_gpu, num_cand_to_retrieve=10):
    # Ensure query_embeddings_batch is numpy array with dtype float32
    assert isinstance(query_embeddings_batch, np.ndarray) and query_embeddings_batch.dtype == np.float32
    print(f"Faiss: query_embeddings_batch.shape: {query_embeddings_batch.shape}")

    # Query the multi-GPU index
    distances, indices = index_gpu.search(query_embeddings_batch, num_cand_to_retrieve)  # (number_of_queries, k)
    return distances, indices

