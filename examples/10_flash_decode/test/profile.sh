#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

python_file=test/test_flash_decode.py
num_gpus=4
rocprof_sys=/opt/rocprofiler-systems/bin/rocprof-sys-run

filename=$(basename "$python_file" .py)
timestamp=$(date +"%Y-%m-%d_%H.%M")
output_base="profile_results/${filename}/${timestamp}"
mkdir -p "$output_base"

# Final outputs
rocprof_output="${output_base}"
output_log_file="${output_base}/${filename}.log"

echo "Rocprof Output Prefix: ${rocprof_output}"

profile=true

# --- FIX 2: Added the --case argument ---
python_cmd="python ${python_file} --case perf"

if [ "$profile" = true ]; then
    echo "[INFO] Running with profiler enabled"
    mpirun --allow-run-as-root -np ${num_gpus} -x PYTHONPATH=. ${rocprof_sys} --use-roctx -o "${rocprof_output}" -- ${python_cmd} 2>&1 | tee "${output_log_file}"
else
    echo "[INFO] Running without profiler"
    mpirun --allow-run-as-root -np ${num_gpus} -x PYTHONPATH=. ${python_cmd} 2>&1 | tee "${output_log_file}"
fi

echo "[INFO] Profiling complete"
echo "[INFO] Output directory: ${rocprof_output}"
echo "[INFO] Output log: ${output_log_file}"