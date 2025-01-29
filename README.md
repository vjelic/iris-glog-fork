# Stream-K + pyrocSHMEM


This repository contains code for experimenting with Stream-K GEMMs + communication kernels.

![dist_gemm](imgs/dist_gemm.excalidraw.svg)


## Algorithms
At the moment we assume that:

$C = A \times B$
where,
* B (weights): sharded column/row-wise across GPUs,
* A (activations): replicated across GPUs,
* C (activations output): replicated across GPUs.

Currently, there are two implementations:

1. Stream-K + All reduce.
Where B is partitioned *row-wise* and hence A is also partitioned column-wise so that we have two tall skinny matrices producing a partial C with shape of M*N and the all reduce kernel reduces the result across all PEs.

2. Stream-K + All scatter
Where B is partitioned  *column-wise* and hence each rank produces non-overlapping columns in the C matrix such that we only need all gather/scatter to broadcast the final result.


## Getting started
An Apptainer definition file that takes care of all depednaices. You can also copy the contents of the `.def` file and create a similar Docker container. We use Apptainer on HPC Fund so we are currently maintaining that. To build the image:
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

Now you are inside a Conda environment. You can run the example code by following these two steps:

```sh
cd stream-k
./shmem_run.sh
```

`./shmem_run.sh` contains the `mpirun` wrapper code and you can edit the script to launch the application for different GEMM and communication variant or different number of  GPUs. GEMM sizes are specified inside the `all_reduce.py` or the `all_scatter.py`.




