num_gpus=8
export OMP_NUM_THREADS=1
collective="all_gather"
# collective="all_reduce"
python -m torch.distributed.run\
        --nproc_per_node=${num_gpus}\
        ${collective}.py