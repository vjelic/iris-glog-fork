import subprocess
import os
from datetime import datetime
import sys

def launch_sbatch(config, m, k, n, num_gpus, algorithm, total_sms, streamk_sms, hash, sbatch_script_content):

    job_name = f"{hash}/{algorithm}_{m}-{k}-{n}_{num_gpus}"

    slurm_out_dir = f"slurm_logs/{job_name}"
    if not os.path.exists(slurm_out_dir):
        os.makedirs(slurm_out_dir, exist_ok=True)
        os.makedirs(slurm_out_dir + "/" + hash , exist_ok=True)

    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")

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
        total_sms = total_sms,
        streamk_sms = streamk_sms,
        hash = hash,
        output_file= os.path.join(
        "../slurm_logs", job_name, f"{job_name}_{timestamp}.json"
    ),
    )

    sbatch_script_path = os.path.join(
        "slurm_logs", job_name, f"{job_name}_{timestamp}.sbatch"
    )
    with open(sbatch_script_path, "w") as sbatch_file:
        sbatch_file.write(formatted_script)
    print(f"SBATCH script saved at: {sbatch_script_path}")

    try:
        subprocess.run(
            ["sbatch", sbatch_script_path], capture_output=True, text=True, check=True
        )
        print(f"Successfully submitted job: {job_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {job_name}")
        print(f"Error message: {e.stderr}")


def main(hashes, config, sbatch_script_content):
    # algorithms = ["all_reduce", "all_scatter"]
    algorithms = ["all_reduce"]
    mnk_array = [
        (4864, 8256, 4096),
        (4096, 8192, 2048),
        (6144, 16384, 8192),
        (8192, 32768, 16384),
        (1024, 4096, 512),
        (2048, 8192, 1024),
        (3072, 12288, 6144),
        (5120, 2048, 1024),
        (16384, 8192, 4096),
        (2560, 10240, 5120)
    ]


    if partition == "mi3008x":
        total_sms = 304
        streamk_sms = 256
    else:
        total_sms = 104
        streamk_sms = 87

    for hash in hashes:
        for algorithm in algorithms:
            for m, n, k in mnk_array:
                max_gpus=8
                min_gpus=1
                num_gpus = min_gpus
                while num_gpus <= max_gpus:
                    launch_sbatch(config, m, k, n, num_gpus,
                                algorithm, total_sms, streamk_sms,
                                hash, sbatch_script_content)
                    num_gpus *= 2


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python main.py <partition> <commit_hash1> <commit_hash2> ...", file=sys.stderr)
        sys.exit(1)

    # partition = "mi2508x"
    # partition = "mi3008x"
    partition = sys.argv[1]

    config = {
        "image_name": "rocshmem_rocm_6.2.3.sif",
        "partition": partition,
        "time_limit": "00:10:00",
        "exclude_list": "",
    }


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
output_file={output_file}
total_sms={total_sms}
streamk_sms={streamk_sms}
hash={hash}
echo "source /opt/conda/bin/activate py_3.10 &&\
    git reset --hard ${{hash}}&&\
    cd stream-k &&\
    mpirun --allow-run-as-root -np ${{num_gpus}}\
        python benchmark.py --algorithm ${{algorithm}}\
            -m ${{m}} -n ${{n}} -k ${{k}}\
                --total_sms ${{total_sms}}\
                --streamk_sms ${{streamk_sms}}\
                --validate --benchmark --debug\
                --output_file ${{output_file}}" \
    | apptainer exec --cleanenv ${{image_path}} bash
    """


    commit_hashes = sys.argv[2:]
    print("Commit Hashes Array:", commit_hashes)
    print("partition:", partition)

    main(commit_hashes, config, sbatch_script_content)