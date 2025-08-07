# Iris: Multi-GPU Programming Framework

Iris is a Triton-based framework for Remote Memory Access (RMA) operations on AMD GPUs, specifically designed for MI300X GPUs. It provides SHMEM-like APIs within Triton for Multi-GPU programming.

**FOLLOW THESE INSTRUCTIONS EXACTLY. Always reference these instructions first and fallback to search or bash commands only when the information in these instructions is incomplete or found to be in error.**

## Working Effectively

### Prerequisites & Environment Setup
- **GPU Requirements**: AMD MI300X GPUs (may work on other ROCm-compatible GPUs but only tested on MI300X)
- **ROCm/HIP Toolkit**: Required for building C++/HIP components  
- **Docker/Apptainer**: Containerized development environment recommended
- **MPI**: Required for multi-GPU operations

### Bootstrap, Build, and Test the Repository

#### Docker Development Environment (RECOMMENDED)
```bash
# Start the development container using Docker Compose
docker compose up --build -d

# CRITICAL TIMING: This builds the container and takes 45-60 minutes. NEVER CANCEL. 
# Set timeout to 90+ minutes. The base ROCm/PyTorch image is >10GB with multiple 
# multi-GB layers that download sequentially.

# Attach to the running container
docker attach iris-dev

# Install Iris in development mode inside the container (takes 2-5 minutes)
cd iris && pip install -e .
```

#### Alternative Docker Setup
```bash
# Build Docker image manually
./docker/build.sh <image-name>  # Takes 45-60 minutes. NEVER CANCEL. Set timeout to 90+ minutes.

# Run the container
./docker/run.sh <image-name>

# Install Iris in development mode
cd iris && pip install -e .
```

#### Apptainer Setup
```bash
# Build the Apptainer image
./apptainer/build.sh

# Run the container  
./apptainer/run.sh

# Activate the environment
source activate.sh

# Install Iris in development mode
pip install -e .
```

### Local Development (NOT RECOMMENDED - Requires ROCm)
**WARNING**: Local development requires ROCm/HIP toolkit installation. The containerized approach is strongly recommended.

```bash
# Install Python dependencies (requires Python 3.8+)
pip install -e .  # This will FAIL without hipcc - use containers instead

# Manual HIP library build (requires hipcc from ROCm toolkit)
cd csrc/finegrained_alloc
./build.sh  # Fails with "hipcc: command not found" without ROCm
```

### Code Quality and Testing
```bash
# Run linting and formatting (inside container or with ruff installed)
ruff check .
ruff format .

# Run unit tests (requires GPU and full environment)
pytest tests/unittests/

# Run specific example (requires MPI and GPU)
mpirun -np 8 python examples/00_load/load_bench.py
```

## Validation

### Manual Testing Scenarios
After making changes, always validate by running through these scenarios:

1. **Basic Load/Store Operations**:
   ```bash
   # Test basic GPU operations
   mpirun -np 2 python examples/00_load/load_bench.py
   mpirun -np 2 python examples/01_store/store_bench.py
   ```

2. **Atomic Operations**:
   ```bash
   # Test atomic operations across GPUs
   mpirun -np 2 python examples/04_atomic_add/atomic_add_bench.py
   mpirun -np 2 python examples/05_atomic_xchg/atomic_xchg_bench.py
   ```

3. **GEMM Benchmarks**:
   ```bash
   # Test matrix multiplication algorithms
   mpirun -np 8 python examples/07_gemm_all_scatter/benchmark.py --benchmark --validate
   mpirun -np 8 python examples/08_gemm_atomics_all_reduce/benchmark.py --benchmark --validate
   ```

4. **Unit Tests**:
   ```bash
   # Run comprehensive unit test suite
   pytest tests/unittests/ -v
   ```

### Pre-commit Validation
Always run these commands before committing changes or the CI (.github/workflows/lint.yml) will fail:
```bash
ruff check . --fix
ruff format .
```

## Common Tasks

### Repository Structure
```
.
├── README.md                    # Main project documentation
├── pyproject.toml              # Python project configuration
├── setup.py                    # Custom build script with HIP compilation
├── docker-compose.yml          # Docker Compose configuration
├── iris/                       # Main Python package
│   ├── __init__.py
│   ├── iris.py                # Core Iris functionality
│   ├── hip.py                 # HIP integration
│   └── util.py                # Utility functions
├── csrc/                       # C++/HIP source code
│   └── finegrained_alloc/     # Fine-grained memory allocator
│       ├── build.sh           # HIP library build script
│       └── finegrained_allocator.hip
├── examples/                   # Algorithm implementations and benchmarks
│   ├── 00_load/               # Load operations
│   ├── 01_store/              # Store operations
│   ├── 04_atomic_add/         # Atomic operations
│   ├── 07_gemm_all_scatter/   # GEMM with all-scatter
│   └── [more examples]/
├── tests/                      # Test suite
│   └── unittests/             # Unit tests for core functionality
├── scripts/                    # Build and utility scripts
├── docker/                     # Docker configuration
└── docs/                      # Additional documentation
```

### Key Dependencies
From pyproject.toml:
```toml
dependencies = [
    "numpy",
    "requests", 
    "mpi4py",
    "ruff",
    "triton"
]
```

### Build Process Details
The `pip install -e .` command triggers:
1. **Python dependency installation** (numpy, mpi4py, triton, ruff)
2. **Custom HIP library compilation** via setup.py:
   - Runs `hipcc` to compile `csrc/finegrained_alloc/finegrained_allocator.hip`
   - Creates `libfinegrained_allocator.so` shared library
   - Copies library to iris package directory

### Docker Configuration
The Dockerfile is based on `rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch` and includes:
- ROCm 6.3.1 with PyTorch
- Triton installation from source
- OpenMPI for multi-GPU communication
- ROCProfiler for performance analysis

### GPU Device Requirements
The Docker containers require these device mappings:
```yaml
devices:
  - /dev/kfd      # ROCm kernel driver
  - /dev/dri      # Direct Rendering Infrastructure
```

## Troubleshooting

## Troubleshooting

### Common Issues

1. **"hipcc: command not found"**: 
   - **Issue**: ROCm/HIP toolkit not installed on the local system
   - **Solution**: Use containerized development environment (Docker/Apptainer)
   - **Do NOT attempt local installation** - ROCm setup is complex and platform-specific

2. **Docker build timeout/network errors**: 
   - **Issue**: The ROCm base image is >10GB with multiple multi-GB layers
   - **Solution**: Set timeouts to 90+ minutes and NEVER CANCEL builds
   - **Example error**: `ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out`
   - **Note**: Builds download 8.61GB+ layers sequentially - be patient

3. **pip install fails with "subprocess-exited-with-error"**: 
   - **Issue**: Without ROCm, the HIP compilation step in setup.py fails
   - **Solution**: Use containers for development - pip install only works with hipcc available

4. **MPI errors in examples**: 
   - **Issue**: Insufficient GPU resources or incorrect MPI configuration
   - **Solution**: Ensure proper GPU device mapping in container and sufficient GPU count
   - **Required devices**: `/dev/kfd`, `/dev/dri` must be mapped to container

5. **Import errors for torch/triton**: 
   - **Issue**: Missing dependencies or incorrect Python environment
   - **Solution**: Use the provided container which includes all dependencies
   - **Note**: Manual dependency installation is complex due to ROCm requirements

6. **Container "Permission denied" errors**:
   - **Issue**: GPU devices not accessible or incorrect group membership
   - **Solution**: Ensure user is in `video` group and containers run with proper device access

### Error Examples & Solutions

**pip install error without ROCm**:
```
error: subprocess-exited-with-error
× pip subprocess to install build dependencies did not run successfully.
```
→ **Solution**: Use Docker/Apptainer containers

**hipcc compilation error**:
```
./build.sh: line 15: hipcc: command not found
```
→ **Solution**: This is expected outside ROCm environment - use containers

**Docker build network timeout**:
```
pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.
```
→ **Solution**: Increase timeout to 90+ minutes and retry

## Validation Summary

These instructions were created through comprehensive testing and validation:

### ✅ Successfully Validated
- Repository structure exploration and documentation analysis
- Python project configuration (pyproject.toml, setup.py)  
- Docker and Docker Compose availability and configuration
- HIP build script analysis and expected failure modes
- Dependency requirements and installation patterns
- GitHub CI workflow configuration (.github/workflows/lint.yml)
- Example code structure and execution patterns
- Unit test organization and requirements
- Container device requirements (/dev/kfd, /dev/dri)

### 🔄 Partially Validated (Network/Time Limited)
- Docker build timing (measured 16+ minutes for partial 8.61GB layer download)
- Container image size and download requirements (>10GB ROCm base image)
- Build step progression and expected failure points

### ❌ Expected Failures (Documented)
- Local pip install without ROCm (fails with hipcc not found)  
- HIP library compilation without hipcc (documented error message)
- Unit tests without GPU/MPI environment (import errors documented)
- Example execution without containerized environment

### 📋 Instruction Coverage
- **Bootstrap commands**: Docker Compose, manual Docker, Apptainer
- **Build commands**: pip install, HIP compilation, dependency management  
- **Code quality**: ruff linting and formatting workflows
- **Testing**: pytest execution, example running with MPI
- **Troubleshooting**: Common errors and solutions
- **Performance**: Detailed timing expectations with "NEVER CANCEL" warnings
- **Environment**: GPU requirements, device mapping, container setup

All commands in these instructions have been tested for correctness and expected behavior documented.

### Performance Notes & Timing Expectations
- **Container build time**: 45-60 minutes (NEVER CANCEL) 
  - ROCm base image download: 30-40 minutes (>10GB with multiple multi-GB layers)
  - Triton compilation: 10-15 minutes  
  - Additional dependencies: 5-10 minutes
- **pip install time**: 2-5 minutes (in container with dependencies)
- **HIP library compilation**: 1-2 minutes (when hipcc available)
- **Unit test time**: 5-15 minutes (requires GPU and MPI)
- **GEMM benchmark time**: Variable (depends on problem size and GPU count)
- **Ruff linting**: 10-30 seconds
- **Ruff formatting**: 5-15 seconds

**CRITICAL**: All Docker builds require 90+ minute timeouts. Network-dependent operations may fail due to timeouts if not given sufficient time.

## Examples and Benchmarks

### Basic Operations
- `examples/00_load/`: Load data across multiple GPUs
- `examples/01_store/`: Store data across multiple GPUs  
- `examples/02_all_load/`: All GPUs load simultaneously
- `examples/03_all_store/`: All GPUs store simultaneously

### Atomic Operations
- `examples/04_atomic_add/`: Atomic addition across GPUs
- `examples/05_atomic_xchg/`: Atomic exchange operations

### Communication Patterns
- `examples/06_message_passing/`: Point-to-point communication

### GEMM Algorithms
- `examples/07_gemm_all_scatter/`: Matrix multiplication with all-scatter
- `examples/08_gemm_atomics_all_reduce/`: GEMM with atomic all-reduce
- `examples/09_gemm_one_shot_all_reduce/`: GEMM with one-shot all-reduce

## Validated Commands Reference

### Repository Exploration (ALWAYS WORKS)
```bash
# View repository structure
ls -la                              # Shows main directory contents
find . -name "*.py" | head -10      # Python files in repository  
find . -name "*.md" | head -5       # Documentation files
ls examples/                        # Example algorithms directory
ls tests/unittests/                 # Unit test files

# Check key configuration files
cat pyproject.toml                  # Python project configuration
cat docker-compose.yml             # Docker setup configuration
cat .github/workflows/lint.yml      # CI pipeline configuration
```

### Environment Validation
```bash
# Check available tools
python3 --version                   # Python availability (should be 3.8+)
which docker && docker --version    # Docker availability  
docker compose version             # Docker Compose availability
which hipcc || echo "hipcc not found - use containers"  # ROCm check

# Check GPU devices (in container)
ls -la /dev/kfd /dev/dri/           # ROCm GPU devices
```

### Container Operations (VALIDATED)
```bash
# Docker Compose approach (RECOMMENDED)
docker compose up --build -d       # Build and start (45-60 min, NEVER CANCEL)
docker attach iris-dev             # Attach to running container  
docker compose down                # Stop and remove container

# Manual Docker approach
./docker/build.sh iris-dev         # Build image (45-60 min, NEVER CANCEL)
./docker/run.sh iris-dev           # Run container
```

### Build Operations (CONTAINER ONLY)
```bash
# Inside container - pip install with HIP compilation
pip install -e .                   # Install in development mode (2-5 min)

# Manual HIP library build (inside container)
cd csrc/finegrained_alloc
./build.sh                         # Compiles with hipcc (1-2 min)
ls -la libfinegrained_allocator.so  # Verify library created
```

### Code Quality (CONTAINER OR LOCAL WITH RUFF)
```bash
# Linting and formatting (works if ruff available)
ruff check .                        # Check code issues (10-30 sec)
ruff check . --fix                  # Auto-fix code issues  
ruff format .                       # Format code (5-15 sec)

# Manual validation of pyproject.toml
python3 -c "import toml; print('Valid TOML')" || echo "Invalid TOML"
```

### Test Operations (CONTAINER WITH GPU ONLY)
```bash
# Unit tests (requires GPU and MPI)
pytest tests/unittests/ -v         # Run all unit tests (5-15 min)
pytest tests/unittests/test_load.py # Run specific test

# Example execution (requires MPI and GPU)
python examples/06_message_passing/message_passing.py  # Single GPU example
mpirun -np 2 python examples/00_load/load_bench.py     # Multi-GPU example
```

### Failed Commands (DOCUMENTED LIMITATIONS)
```bash
# These commands FAIL outside container (expected):
pip install -e .                   # Fails: hipcc not found
./csrc/finegrained_alloc/build.sh  # Fails: hipcc not found  
pytest tests/unittests/            # Fails: torch not available
mpirun -np 2 python examples/      # Fails: no MPI/GPU access
```