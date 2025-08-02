# Distributed Flash Decode with Iris

It uses a tensor-parallel strategy where the Key/Value (KV) cache is sharded across multiple GPUs. Iris is used directly in fused Triton Kernels for overlapping computation/communication.

---
## Testing and Benchmarking

Use `mpirun` to launch the test scripts on multiple GPUs.

* **Main Test Script**: `test/test_flash_decode.py` is the primary script for testing all Iris and MPI-based implementations.

* **Correctness Test**: Verifies the output against a reference implementation. Set `<num_gpus>` to the number of GPUs you want to run on.
    ```bash
    mpirun -np <num_gpus> python examples/10_flash_decode/test/test_sp_fd.py --case correctness
    ```

* **Performance Comparison**: Benchmarks all versions and writes results to `compare_results.json`. This is the primary script for evaluation.
    ```bash
    mpirun -np <num_gpus> python examples/10_flash_decode/test/test_sp_fd.py --case compare
    ```
* For the RCCL test, use the `test/test_flash_decode_rccl.py` file with the same `--case` flags, using `torchrun`.

---
## Implementations

Different layers explore various levels of kernel fusion and communication.

* **`fd_layer_iris.py`**: A standard two-step approach: first, it combines local results, then it uses a generic Iris All-Gather to communicate.
* **`fd_layer_iris_fused.py`**: Fuses the intra-rank combination with the inter-rank data push into a single kernel, using coarse-grained atomics for synchronization.
* **`fd_layer_iris_fused_full.py`**: The most optimized version, using fine-grained, per-tile signaling for better parallelism during the final combination step.
* **`fd_layer_rccl.py` / `fd_layer_mpi.py`**: Baseline implementations using RCCL and MPI for performance comparison.

All Triton attention kernels are located in `kernels/decode_kernels.py`.


---
## 📂 File Structure

* **`kernels/`**: This folder contains the low-level Triton kernels (.py files with `@triton.jit` functions) that perform the core attention calculations and communication logic.

* **`layers/`**: This folder contains high-level PyTorch modules (torch.nn.Module) that wrap the Triton kernels. Each file represents a different distributed strategy for Flash Decode.

* **`test/`**: This folder holds all scripts used for verifying the correctness and benchmarking the performance of the different layers.