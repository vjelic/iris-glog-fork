#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# Get the directory where the script itself is located
python_file="test/test_flash_decode.py"
num_gpus=8
rocprof_sys=/opt/rocprofiler-systems/bin/rocprof-sys-run

filename=$(basename "$python_file" .py)
output_base="profile_results/${filename}"
mkdir -p $output_base

# Final outputs
output_log_file="${output_base}.log"
rocprof_output="${output_base}"

echo "Rocprof Output Prefix: ${rocprof_output}"

profile=true

python_cmd="python ${python_file}"


if [ "$profile" = true ]; then
    echo "[INFO] Running with profiler enabled"

    mpirun --allow-run-as-root -np ${num_gpus} ${rocprof_sys} \
    -T \
    --perfetto-annotations \
    --output "${rocprof_output}" \
    --use-rocm \
    --rocm-marker-api-operations-annotate-backtrace roctxGetThreadId roctxMarkA roctxRangePop roctxRangePushA roctxRangeStartA roctxRangeStop \
    -- ${python_cmd} --case perf  2>&1 | tee ${rocprof_output}/${output_log_file}
else
    echo "[INFO] Running without profiler"
    mpirun --allow-run-as-root -np ${num_gpus} ${python_cmd}  2>&1 | tee ${output_log_file}
fi