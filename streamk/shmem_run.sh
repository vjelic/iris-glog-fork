num_gpus=8
export OMP_NUM_THREADS=1
# collective="all_gather"
app="shmem_distributed_gemm"

# TRITON_INTERPRET=1
mpirun -np $num_gpus python ${app}.py