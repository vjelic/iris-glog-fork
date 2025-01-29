FROM rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch
  
# Use bash shell for RUN commands  
SHELL ["/bin/bash", "-c"]  
  
# Set environment variables  
ENV TRITON_PATH=/workspace/triton \  
    ROCM_PATH=/opt/rocm \  
    OMPI_MCA_mtl="^ofi" \  
    OMPI_MCA_pml="ob1" \  
    PYTHONPATH=/workspace/triton/python/ \  
    LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/openmpi/lib:$LD_LIBRARY_PATH  
  
# Install git and other dependencies  
RUN apt-get update && \  
    apt-get install -y git wget  
  
# Source conda.sh for conda commands  
ENV CONDA_SOURCE="/opt/conda/etc/profile.d/conda.sh"  
  
# Check available conda environments  
RUN source $CONDA_SOURCE && conda env list  
  
# Create the py3.10 conda environment if it doesn't exist  
RUN source $CONDA_SOURCE && \  
    if ! conda env list | grep -q 'py_3.10'; then \  
        conda create -y -n py_3.10 python=3.10; \  
    fi  
  
# Activate conda environment and install packages  
RUN source $CONDA_SOURCE && \  
    conda activate py_3.10 && \  
    conda install -y -c conda-forge mpi4py openmpi jupyter ninja cmake wheel  
  
# Set PATH to include the conda environment's bin directory  
ENV PATH="/opt/conda/envs/py_3.10/bin:$PATH"  
  
# Clone Triton and install it  
RUN source $CONDA_SOURCE && \  
    conda activate py_3.10 && \  
    git clone https://github.com/triton-lang/triton.git $TRITON_PATH && \  
    cd $TRITON_PATH && \  
    pip install -e python  
  
# Install rocprofiler-systems  
RUN wget https://github.com/ROCm/rocprofiler-systems/releases/latest/download/rocprofiler-systems-install.py && \  
    python3 ./rocprofiler-systems-install.py --prefix /opt/rocprofiler-systems --rocm 6.3 && \  
    rm -f rocprofiler-systems-install.py  
  
# Update LD_LIBRARY_PATH and PATH  
ENV LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/openmpi/lib:$LD_LIBRARY_PATH" \  
    PATH="/opt/conda/envs/py_3.10/bin:/opt/rocm/bin:/usr/lib/openmpi/bin:$PATH"  
  
# Create the entrypoint script  
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'echo "Welcome to the ROCm-aware Docker image!"' >> /entrypoint.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /entrypoint.sh && \
    echo 'conda activate py_3.10' >> /entrypoint.sh && \
    echo 'export PATH=$(conda info --base)/envs/py_3.10/bin:$PATH' >> /entrypoint.sh && \
    echo 'if [ $# -eq 0 ]; then' >> /entrypoint.sh && \
    echo '    exec /bin/bash' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '    exec "$@"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh  

# Set the entrypoint  
ENTRYPOINT ["/bin/bash", "-c", "source /entrypoint.sh && exec bash"]