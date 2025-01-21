from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data to send: each process creates a list of size elements
send_data = [rank + i for i in range(size)]
print(f"Process {rank} is sending: {send_data}")

# Prepare a buffer to receive data from all processes
recv_data = [None] * size

# Perform all-to-all communication
comm.Alltoall(send_data, recv_data)

# Print received data
print(f"Process {rank} received: {recv_data}")
