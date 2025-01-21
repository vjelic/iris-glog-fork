import numpy as np
from mpi_helpers import mpi_allgather
from mpi4py import MPI

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.array([rank], dtype="int")
    print(f"Process {rank} initial data: {data}")
    gathered_data = mpi_allgather(data)
    print(f"Process {rank} gathered data: {gathered_data}")
