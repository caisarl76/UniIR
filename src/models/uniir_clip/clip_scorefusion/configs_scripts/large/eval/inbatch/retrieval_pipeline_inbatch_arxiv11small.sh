#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Initialize Conda
source /opt/conda/etc/profile.d/conda.sh # <--- Change this to the path of your conda.sh

# Path to the codebase and config file
SRC="$HOME/uniir/src"  # Absolute path to codebse /UniIR/src # <--- Change this to the path of your UniIR/src

# Path to common dir
COMMON_DIR="$SRC/common"

# Path to MBEIR data and MBEIR directory where we store the checkpoints, embeddings, etc.
UNIIR_DIR="/root/uniir/" # <--- Change this to the MBEIR directory
MBEIR_DATA_DIR="/data/multimodal/arxiv_qa/" # <--- Change this to the MBEIR data directory you download from HF page

# Path to config dir
MODEL="uniir_clip/clip_scorefusion"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="eval"  # <--- Change this to the mode you want to run
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs_scripts/$SIZE/$MODE/$EXP_NAME"

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=1  # <--- Change this to the CUDA devices you want to use
NPROC=1 # <--- Change this to the number of GPUs you want to use
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo  "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# CD to script directory
cd $COMMON_DIR

# Activate conda environment
# conda activate clip
source activate faiss # <--- Change this to the name of your conda environment

# Run retrieval command
CONFIG_PATH="$CONFIG_DIR/retrieval_arxivqa11small.yaml"
SCRIPT_NAME="mbeir_retriever.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

python $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --enable_retrieval