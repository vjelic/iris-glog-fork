#!/bin/bash

num_gpus=8
algorithm=one_shot_v1
m=8192
n=4608
k=36864
total_sms=304
streamk_sms=256
blk_m=256
blk_n=256
blk_k=64
gsize_m=6
COMMUNICATION_TILE_M=128
COMMUNICATION_TILE_N=64
num_stages=2
num_warps=8
communication_block_size=256
datatype="fp16"

timestamp=$(date +"%Y%m%d_%H%M%S")

mkdir -p $timestamp

# Build a descriptive base name
output_base="${timestamp}/fp16_comm${COMMUNICATION_TILE_M}x${COMMUNICATION_TILE_N}x${communication_block_size}_${m}_${n}_${k}_streamksms${streamk_sms}_blk_m${blk_m}_blk_n${blk_n}_blk_k${blk_k}_gsize_m${gsize_m}"

# Final outputs
output_json_file="${output_base}.json"
output_log_file="${output_base}.log"
rocprof_output="${output_base}-%pid%"

# Echo for verification
echo "JSON Output: ${output_json_file}"
echo "Rocprof Output Prefix: ${rocprof_output}"

profiler_enabled=false 

python_cmd="python benchmark.py \
    --algorithm ${algorithm} \
    -m ${m} -n ${n} -k ${k} \
    --total_sms ${total_sms} \
    --gemm_sms ${streamk_sms} \
    --BLK_M ${blk_m} --BLK_N ${blk_n} \
    --BLK_K ${blk_k} --gsize_m ${gsize_m} \
    --validate --benchmark --debug \
    --heap_size 8589934592 \
    --COMMUNICATION_TILE_M ${COMMUNICATION_TILE_M} \
    --COMMUNICATION_TILE_N ${COMMUNICATION_TILE_N} \
    --output_file ${output_json_file} \
    --num_stages ${num_stages} \
    --num_warps ${num_warps} \
    --datatype ${datatype}\
    --communication_block_size ${communication_block_size}"

# Launch
if [ "$profiler_enabled" = true ]; then
    echo "[INFO] Running with profiler enabled"
    mpirun --allow-run-as-root -np ${num_gpus} /opt/rocprofiler-systems/bin/rocprof-sys-run \
        --use-roctx -o "${rocprof_output}" \
        -- ${python_cmd}
else
    echo "[INFO] Running without profiler"
    mpirun --allow-run-as-root -np ${num_gpus} ${python_cmd}  2>&1 | tee ${output_log_file}
fi
