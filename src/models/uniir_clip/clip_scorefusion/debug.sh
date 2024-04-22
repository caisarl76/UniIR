CUDA_VISIBLE_DEVICES=1 python train_arxiv_debug.py --config_path /root/uniir/src/models/uniir_clip/clip_scorefusion/configs_scripts/large/train/inbatch/inbatch_arxivqa12.yaml --uniir_dir /root/uniir/ --mbeir_data_dir /data/multimodal/arxiv_qa --dataNum 11 --train_size 90000 --val_size 2000

CUDA_VISIBLE_DEVICES=1 python train_arxiv_debug.py --config_path /root/uniir/src/models/uniir_clip/clip_scorefusion/configs_scripts/large/train/inbatch/inbatch_arxivqa12.yaml --uniir_dir /root/uniir/ --mbeir_data_dir /data/multimodal/arxiv_qa --dataNum 12 --train_size 90000 --val_size 2000
