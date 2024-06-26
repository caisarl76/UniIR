# Train CLIPScoreFusion model on MBEIR dataset

# Initialize Conda
source /opt/conda/etc/profile.d/conda.sh # <--- Change this to the path of your conda.sh

# Path to the codebase and config file
SRC="$HOME/uniir/src"  # Absolute path to codebse /UniIR/src # <--- Change this to the path of your UniIR/src

# Path to common dir
COMMON_DIR="$SRC/common"

# Path to MBEIR data and UniIR directory where we store the checkpoints, embeddings, etc.
UNIIR_DIR="/root/uniir/" # <--- Change this to the UniIR directory
MBEIR_DATA_DIR="/data/multimodal/arxiv_qa/" # <--- Change this to the MBEIR data directory you download from HF page

# Path to config dir
# MODEL="Bingsu/clip-vit-large-patch14-ko"  # <--- Change this to the model you want to run
MODEL="uniir_clip/clip_scorefusion"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="train"  # <--- Change this to the mode you want to run
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs_scripts/$SIZE/$MODE/$EXP_NAME"

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=3 # <--- Change this to the CUDA devices you want to us
NPROC=1
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo  "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Update config
CONFIG_PATH="$CONFIG_DIR/inbatch_arxivqa11.yaml"
cd $COMMON_DIR
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

# Change to model directory
cd $MODEL_DIR
SCRIPT_NAME="train.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Activate conda environment
# conda activate clip
source activate clip # <--- Change this to the name of your conda environment

# Run training command
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port 29501 $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"