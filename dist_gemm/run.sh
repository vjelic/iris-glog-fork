num_gpus=2
export OMP_NUM_THREADS=1
# collective="all_gather"
collective="all_reduce"
# python -m torch.distributed.run --nproc_per_node=${num_gpus} ${collective}_dist_gemm_B.py
python -m torch.distributed.run --nproc_per_node=${num_gpus} ${collective}.py