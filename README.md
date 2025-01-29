# Stream-K + pyrocSHMEM


This repository contains code for experimenting with Stream-K GEMMs + communication kernels.

![dist_gemm](imgs/dist_gemm.excalidraw.svg)


## Algorithms
At the moment we assume that:

$C = A \times B$
where,
* $B$ (weights): sharded column/row-wise across GPUs,
* $A$ (activations): replicated across GPUs, and
* $C$ (activations output): replicated across GPUs.

Currently, there are two implementations:

1. Stream-K + All reduce.
Where $B$ is partitioned *row-wise* and hence $A$ is partitioned column-wise so that we have two tall skinny matrices producing a partial $C$ with shape of $M \times N$ and the all reduce kernel reduces the results across all GPUs or ranks (right figure).

2. Stream-K + All scatter
Where $B$ is partitioned  *column-wise* and hence each rank produces non-overlapping columns in the output $C$ matrix such that we only need all gather/scatter to broadcast the final result (left figure).


## Getting started
We provide an Apptainer definition file contains all depednaices to reproduce the result. You can also copy the contents of the `.def` file and create a similar Docker container. We use Apptainer on HPC Fund so we are currently maintaining that. To build the image:
```shell
./apptainer/build.sh
```

And then to run the image,
```shell
./apptainer/run.sh
```

Once inside the Apptainer image, source the `activate.sh` script.

```
source activate.sh
```

Now you are inside a Conda environment. Before running code, you need to build `finegrained_alloc`, a C library interface for fine-grained allocation. The plugin is required to redirect PyTorch allocation to fine-grained memory.

```shell
cd pyrocSHMEM/finegrained_alloc/
./build.sh
cd ../../
```


You can run the example code by following these two steps:

```shell
cd stream-k
./run.sh
```

```terminal
(py_3.10) Apptainer> ./run.sh --help
Usage: ./run.sh [OPTIONS]

Options:
  -n, --num-gpus NUM      Number of GPUs to use (default: 8)
  -c, --collective NAME   Collective operation to run (default: all_scatter)
  -h, --help              Show this help message and exit

Example:
  ./run.sh -n 4 -c all_reduce  # Run with 4 GPUs using all_reduce
```


### Reference implementations

There are two reference implementations (`all_gather.py` and `all_reduce.py`) that use RCCL inside the [reference](./reference/) directory. To run any of them,

```shell
cd reference
./run.sh
```

```terminal
(py_3.10) Apptainer> ./run.sh --help
Usage: ./run.sh [OPTIONS]

Options:
  -n, --num-gpus NUM      Number of GPUs to use (default: 2)
  -c, --collective NAME   Collective operation to run (default: all_gather)
  -h, --help              Show this help message and exit

Example:
  ./run.sh -n 4 -c all_reduce  # Run with 4 GPUs using all_reduce
```