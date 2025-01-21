# ROCSHMEM_INSTALL_DIR="/home1/muhaawad/rocshmem/"
# OPENMPI_UCX_INSTALL_DIR="/usr/local/ompi/"
# hipcc -c -fgpu-rdc -x hip rocshmem_allreduce_test.cc \
#   --offload-arch=gfx90a \
#   -I/opt/rocm/include \
#   -I$ROCSHMEM_INSTALL_DIR/include \
#   -I$OPENMPI_UCX_INSTALL_DIR/include/

# hipcc -fgpu-rdc --hip-link rocshmem_allreduce_test.o -o rocshmem_allreduce_test\
#   --offload-arch=gfx90a \
#   $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a \
#   $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so \
#   -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

UCX_NET_DEVICES=eth0 ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 8 ./rocshmem_allreduce_test