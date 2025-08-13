# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import subprocess
import os
from datetime import datetime
import argparse
import json


def launch_sbatch(
    config,
    m,
    k,
    n,
    num_gpus,
    algorithm,
    gemm_sms,
    blk_m,
    blk_n,
    blk_k,
    gsize_m,
    hash,
    python_file,
    sbatch_script_content,
    dry_run=False,
):
    job_name = f"{hash}/{algorithm}_{m}-{k}-{n}_{num_gpus}"

    slurm_out_dir = f"slurm_logs/{job_name}"
    if not os.path.exists(slurm_out_dir):
        os.makedirs(slurm_out_dir, exist_ok=True)
        os.makedirs(slurm_out_dir + "/" + hash, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    output_json = os.path.abspath(os.path.join("slurm_logs", job_name, f"{job_name}_{timestamp}.json"))
    output_log = os.path.abspath(os.path.join("slurm_logs", job_name, f"{job_name}_{timestamp}.log"))

    formatted_script = sbatch_script_content.format(
        job_name=job_name,
        image_name=config["image_name"],
        partition=config["partition"],
        m=m,
        n=n,
        k=k,
        num_gpus=num_gpus,
        algorithm=algorithm,
        time_limit=config["time_limit"],
        exclude_list=config["exclude_list"],
        gemm_sms=gemm_sms,
        blk_m=blk_m,
        blk_n=blk_n,
        blk_k=blk_k,
        gsize_m=gsize_m,
        hash=hash,
        output_json_file=output_json,
        output_log_file=output_log,
        python_file=python_file,
    )

    sbatch_script_path = os.path.join("slurm_logs", job_name, f"{job_name}_{timestamp}.sbatch")
    with open(sbatch_script_path, "w") as sbatch_file:
        sbatch_file.write(formatted_script)
    print(f"SBATCH script saved at: {sbatch_script_path}")
    print(f"Output JSON at: {output_json}")
    print(f"Output log at: {output_log}")

    if dry_run:
        return

    try:
        if config["partition"] is None:
            os.chmod(sbatch_script_path, 0o755)
            subprocess.run(
                sbatch_script_path,
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            subprocess.run(
                ["sbatch", sbatch_script_path],
                capture_output=True,
                text=True,
                check=True,
            )

        print(f"Successfully submitted job: {job_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {job_name}")
        print(f"Error message: {e.stderr}")


def main(hashes, config, sbatch_script_content, input_json, tiling_json, dry_run):
    algorithms = ["all_reduce", "all_scatter", "one_shot"]

    with open(input_json, "r") as file:
        data = json.load(file)

    with open(tiling_json, "r") as file:
        tiling_data = json.load(file)

    unique_mkn = list(set((entry["m"], entry["k"], entry["n"]) for entry in data))

    optional_keys = ["BLK_M", "BLK_N", "BLK_K", "gsize_m"]
    mkn_gemm_tiles = {}

    for entry in tiling_data:
        mkn = (entry["m"], entry["k"], entry["n"])
        if mkn not in mkn_gemm_tiles:
            mkn_gemm_tiles[mkn] = {key: entry[key] for key in optional_keys if key in entry}

    if config["partition"] is not None:
        if "mi300" in config["partition"]:
            print("Running on MI300")
            gemm_sms = 304
        elif "mi250" in config["partition"]:
            print("Running on MI250")
            gemm_sms = 104
    else:
        print("Assuming MI300")
        gemm_sms = 304

    enable_algorithms = False
    enable_mkn = True

    algorithms_iter = algorithms if enable_algorithms else ["all_scatter"]
    unique_mkn_iter = list(enumerate(unique_mkn)) if enable_mkn else [(0, (8192, 36864, 4608))]

    #python_file = "examples/07_gemm_all_scatter/benchmark.py"
    python_file = "examples/10_gemm_all_scatter_wg_specialization/benchmark.py"
    python_file = "examples/12_gemm_all_scatter_bulk_synchronous/benchmark.py"
    for hash in hashes:
        for algorithm in algorithms_iter:
            for i, (m, k, n) in unique_mkn_iter:
                max_gpus = 8
                min_gpus = 1
                num_gpus = min_gpus
                print(f"Index: {i} / {len(unique_mkn)}, m: {m}, k: {k}, n: {n}")
                while num_gpus <= max_gpus:
                    # Figure out the magic tile sizes
                    key = None
                    if algorithm in ("all_reduce", "one_shot"):
                        key = (m, k // num_gpus, n)
                    elif algorithm == "all_scatter":
                        key = (m, k, n // num_gpus)
                    # Check for missing entry
                    if key not in mkn_gemm_tiles:
                        print(f"[WARNING] GEMM params not found for {algorithm} with key={key}, using default.")
                    # Now safely get the params
                    gemm_params = mkn_gemm_tiles.get(
                        key,
                        {
                            "BLK_M": 256,
                            "BLK_N": 256,
                            "BLK_K": 32,
                            "gsize_m": 8,
                        },
                    )
                    # Extract values
                    blk_m = gemm_params.get("BLK_M")
                    blk_n = gemm_params.get("BLK_N")
                    blk_k = gemm_params.get("BLK_K")
                    gsize_m = gemm_params.get("gsize_m")

                    if blk_m == 128 and blk_n == 256:
                        blk_m = 256
                        blk_n = 128
                    if blk_m == 128 and blk_n == 128:
                        blk_m = 256
                        blk_n = 128

                    launch_sbatch(
                        config,
                        m,
                        k,
                        n,
                        num_gpus,
                        algorithm,
                        gemm_sms,
                        blk_m,
                        blk_n,
                        blk_k,
                        gsize_m,
                        hash,
                        python_file,
                        sbatch_script_content,
                        dry_run=dry_run,
                    )
                    num_gpus *= 2


if __name__ == "__main__":
    iris_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="Process partition and commit hashes.")
    parser.add_argument("--partition", nargs="?", default=None, help="The partition name (optional)")
    parser.add_argument("--commit_before", nargs="?", default=None, help="Commit hash before (optional)")
    parser.add_argument("--commit_after", nargs="?", default=None, help="Commit hash after (optional)")
    parser.add_argument("--input_json", type=str, default=os.path.join(iris_dir, "dataset", "mini.json"), help="Path to input JSON file")
    parser.add_argument("--tiling_json", type=str, default=os.path.join(iris_dir, "dataset", "tiling.json"), help="Path to input JSON file")
    parser.add_argument(
        "--dry_run",
        "-n",
        action="store_true",
        help="dry_run run (do not execute any commands)",
    )
    parser.add_argument(
        "--apptainer",
        action="store_true",
        help="use apptainer container (default: assume already inside container)",
    )
    

    args = parser.parse_args()
    partition = args.partition

    commit_before = args.commit_before or "latest"
    commit_after = args.commit_after or "latest"

    commit_hashes = list(dict.fromkeys([commit_before, commit_after]))

    config = {
        "image_name": "iris.sif",
        "partition": partition,
        "time_limit": "00:05:00",
        "exclude_list": "",
        "use_apptainer": args.apptainer,
    }

    # Create sbatch script content based on whether to use apptainer
    if config["use_apptainer"]:
        sbatch_script_content = """#!/bin/bash
#SBATCH -J {job_name}                               # Job name
#SBATCH -o slurm_logs/{job_name}/{job_name}.%j.out  # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                                        # Total number of nodes requested
#SBATCH -n 128                                      # Total number of mpi tasks requested
#SBATCH -t {time_limit}                             # Run time (hh:mm:ss)
#SBATCH --partition={partition}                     # Partition
#SBATCH --exclude={exclude_list}                    # Exclude list (e.g., node[01,13-15])
image_path=./apptainer/images/{image_name}
num_gpus={num_gpus}
algorithm={algorithm}
m={m}
n={n}
k={k}
output_json_file={output_json_file}
output_log_file={output_log_file}
gemm_sms={gemm_sms}
blk_m={blk_m}
blk_n={blk_n}
blk_k={blk_k}
gsize_m={gsize_m}
python_file={python_file}
num_stages=2
datatype=fp16
num_warps=8
hash={hash}
echo "source /opt/conda/bin/activate py_3.10 &&\
    if [ \"${{hash}}\" != \"latest\" ]; then \
        git reset --hard ${{hash}}; \
    fi && \
    pip install -e . && \
    timeout 5m mpirun --allow-run-as-root -np ${{num_gpus}}\
        python ${{python_file}}\
            -m ${{m}} -n ${{n}} -k ${{k}}\
                --gemm_sms ${{gemm_sms}}\
                --BLK_M ${{blk_m}}\
                --BLK_N ${{blk_n}}\
                --BLK_K ${{blk_k}}\
                --gsize_m ${{gsize_m}}\
                --validate --benchmark --debug\
                --heap_size 8589934592\
                --output_file ${{output_json_file}}\
                --datatype ${{datatype}}
        &> $output_log_file" \
    | apptainer exec --cleanenv ${{image_path}} bash
    """
    else:
        sbatch_script_content = """#!/bin/bash
#SBATCH -J {job_name}                               # Job name
#SBATCH -o slurm_logs/{job_name}/{job_name}.%j.out  # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                                        # Total number of nodes requested
#SBATCH -n 128                                      # Total number of mpi tasks requested
#SBATCH -t {time_limit}                             # Run time (hh:mm:ss)
#SBATCH --partition={partition}                     # Partition
#SBATCH --exclude={exclude_list}                    # Exclude list (e.g., node[01,13-15])
num_gpus={num_gpus}
algorithm={algorithm}
m={m}
n={n}
k={k}
output_json_file={output_json_file}
output_log_file={output_log_file}
gemm_sms={gemm_sms}
blk_m={blk_m}
blk_n={blk_n}
blk_k={blk_k}
gsize_m={gsize_m}
python_file={python_file}
num_stages=2
datatype=fp16
num_warps=8
hash={hash}
source /opt/conda/bin/activate py_3.10
if [ "${{hash}}" != "latest" ]; then \
    git reset --hard ${{hash}}; \
fi
pip install -e .
timeout 5m mpirun --allow-run-as-root -np ${{num_gpus}}\
    python ${{python_file}}\
        -m ${{m}} -n ${{n}} -k ${{k}}\
            --BLK_M ${{blk_m}}\
            --BLK_N ${{blk_n}}\
            --BLK_K ${{blk_k}}\
            --gsize_m ${{gsize_m}}\
            --validate --benchmark --debug\
            --heap_size 8589934592\
            --output_file ${{output_json_file}}\
            --datatype ${{datatype}}
    &> $output_log_file
    """

    main(
        commit_hashes,
        config,
        sbatch_script_content,
        args.input_json,
        args.tiling_json,
        args.dry_run,
    )
