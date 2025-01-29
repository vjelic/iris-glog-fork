num_gpus=8
export OMP_NUM_THREADS=1


collective="all_scatter"
# collective="all_reduce"

# TRITON_INTERPRET=1
mpirun -np $num_gpus python ${collective}.py