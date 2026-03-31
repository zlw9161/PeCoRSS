# Single mode
# > CUDA_VISIBLE_DEVICES=4 python run.py --cfg config_files/pkcin4cwr.json

# DP mode
# e.g., config files: gpu_id=[0,1], device="cuda:0"
# > python run_dp.py --cfg config_files/pkcin4cwr.json

# DDP mode
# e.g. windows w/ gloo backend: python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 --master_port=12355 run.py --cfg config_files/pkcin4cwr.json
# --nnodes sever --nproc_per_node gpu
# ps: config file device must be cuda, e.g. device="cuda"
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=12355 run.py --cfg config_files/pkcin4any_cuda.json

## echmo case
# OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=12355 run_echmo.py --cfg config_files/echmo.json
# OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=12355 run_echmo.py --cfg config_files/echmo.json --profile
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1,2 torchrun --nnodes=1 --nproc_per_node=2 --master_port=12355 run_echmo.py --cfg config_files/echmo_itc.json