"""
Training Code for CLIP-SF
"""

# Standard library
import sys
sys.path.append('/root/uniir/src/')
import argparse
import logging
import os
import random

# Third-party
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb

# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_dataloader_list,
)
from models.uniir_clip.engine import train_one_epoch, eval_engine
from models.uniir_clip import utils
from clip_sf import CLIPScoreFusion
from custom.hf_clip import HF_CLIPScoreFusion
from transformers import AutoConfig, AutoProcessor
# Set up logger
logger = logging.getLogger()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def filter_parameters(model, condition_fn):
    named_parameters = model.named_parameters()
    return [p for n, p in named_parameters if condition_fn(n, p) and p.requires_grad]


def create_optimizer(gain_or_bias_params, rest_params, config):
    return optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=config.trainer_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )


def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config):
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    save_obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict(),
    }
    checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def log_results(train_stats, val_stats, test_stats, epoch=None, best_epoch=None):
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats


def train(
    train_loader,
    val_loader,
    model,
    model_without_ddp,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch,
):
    global_step, total_loss, best_inbatch_accuracy = (
        0,
        0.0,
        0.0,
    )  # TODO: global_step is not used.
    best_epoch = 0
    model.zero_grad()
    gpu_id='cuda'
    
    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        # Set different seed for different epoch

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            gpu_id,
            scheduler,
            global_step,
            scaler,
            config,
        )

        eval_freq = config.evaluator.eval_freq
        if val_loader is None or epoch % eval_freq != 0:
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
        else:
            val_status = eval_engine(model, val_loader, gpu_id, config)
            try:
                inbatch_accuracy = float(val_status["inbatch_accuracy"])
            except ValueError:
                print(f"Error: Expected a number but got '{val_status['inbatch_accuracy']}'")
                inbatch_accuracy = 100.0
            # Note: still save the model even if the in-batch accuracy is not the best
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
            if inbatch_accuracy >= best_inbatch_accuracy:
                # if utils.is_main_process():
                #     save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
                best_inbatch_accuracy = inbatch_accuracy
                best_epoch = epoch

        torch.cuda.empty_cache()


def main(config):
    # Set up seed for reproducibility
    cudnn.benchmark = True

    # Initialize and load model
    print("Creating CLIP-SF model...")
    model_config = config.model
    pretrained_clip_model_dir = os.path.join(config.uniir_dir, model_config.pretrained_clip_model_dir)
    logger.info(f"Downloading CLIP model to {pretrained_clip_model_dir}...")
    
    cfg = AutoConfig.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
    model = HF_CLIPScoreFusion(cfg)
    processor = AutoProcessor.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
    img_preprocess_fn = processor.image_processor
    tokenizer = processor.tokenizer
    
    # Set up optimizer, and scaler
    # Apply different optimization strategies to different parameters
    # This is adapted from the UniVL-DR codebase
    exclude_condition = lambda n, p: p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])
    include_condition = lambda n, p: not exclude_condition(n, p)
    gain_or_bias_params = filter_parameters(model, exclude_condition)
    rest_params = filter_parameters(model, include_condition)
    optimizer = create_optimizer(gain_or_bias_params, rest_params, config)
    scaler = GradScaler()  # Initialize the GradScaler

    # If resume training, load the checkpoint

    # Move model to GPUs
    model.train()
    model = model.to('cuda')
    model_without_ddp = model

    # Prepare datasets and dataloaders
    logger.info("Preparing dataset ...")  # Note printing only available in the main process
    logger.info(f"Loading dataset from {config.mbeir_data_dir}{config.data_config.train_query_data_path}...")

    # img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()

    train_dataset, train_collector = build_mbeir_dataset_from_config(
        config=config,
        tokenizer=tokenizer,
        img_preprocess_fn=img_preprocess_fn,
        dataset_type=DatasetType.MAIN_TRAIN,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader_config.train_batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        shuffle=False,  # Note: since we use sampler, shuffle should be False
        collate_fn=train_collector,
        drop_last=True,
    )

    enable_eval = config.evaluator.enable_eval
    valid_loader = None
    if enable_eval:
        in_batch_val_dataset, in_batch_val_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.IN_BATCH_VAL,
        )
        valid_loader = DataLoader(
            dataset=in_batch_val_dataset,
            batch_size=config.dataloader_config.valid_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=True,
            shuffle=False,  # Note: since we use sampler, shuffle should be False
            collate_fn=in_batch_val_collector,
            drop_last=True,
        )
    else:
        print("In-batch validation is disabled.")

    # Initializing the scheduler
    t_total = (
        len(train_loader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=0)

    epoch = 0


    # Training loop
    train(
        train_loader,
        valid_loader,
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--uniir_dir",
        type=str,
        default="/data/UniIR",
        help="Path to UniIR directory to save checkpoints, embeddings, etc.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data",
        help="Path to mbeir dataset directory",
    )
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument(
    #     "--gpu",
    #     type=int,
    #     default=0
    # )
    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir


    # Set up logger
    
    logger_out_dir = os.path.join(config.uniir_dir, config.logger_config.logger_out_dir)
    logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
    if not os.path.exists(logger_out_dir):
        os.makedirs(logger_out_dir, exist_ok=True)
    handlers = [logging.FileHandler(logger_out_path), logging.StreamHandler()]
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.DEBUG,
        datefmt="%d-%m-%Y %H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.info(config)

    main(config)

