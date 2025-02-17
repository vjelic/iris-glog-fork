# export TRITON_INTERPRET=1





mpirun --allow-run-as-root -np 1 python benchmark.py\
        --algorithm one_shot\
        -m 4096 -k 11008 -n 65536\
        --validate --debug\
        --benchmark\
        --trace_tiles\
        --output_file perf.log

# Perf
# mpirun --allow-run-as-root -np 1 python benchmark.py\
#         --algorithm all_reduce\
#         -m 4096 -k 11008 -n 65536\
#         --total_sms 304  --streamk_sms 200\
#         --validate --debug\
#         --benchmark\
#         --communication_block_size 256\
#         --COMMUNICATION_TILE_M 256\
#         --COMMUNICATION_TILE_N 256\
#         --output_file perf.log

# Bug

# mpirun --allow-run-as-root -np 2 python benchmark.py\
#         --algorithm all_reduce\
#         -m 512 -n 256 -k 512\
#         --total_sms 2  --streamk_sms 1\
#         --BLK_M 512 --BLK_N 512 --BLK_K 32\
#         --COMMUNICATION_TILE_M 512\
#         --COMMUNICATION_TILE_N 512\
#         --communication_block_size 32\
#         --validate --debug\
#         --output_file bad.log