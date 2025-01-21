import ctypes
import numpy as np

class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def print_reserved_field(ipc_handle_struct, rank=None):
    """
    Print the contents of the `reserved` field of `hipIpcMemHandle_t` in hexadecimal.

    Args:
        ipc_handle_struct: A `hipIpcMemHandle_t` instance.
        rank: Optional rank for debugging output.
    """
    if not isinstance(ipc_handle_struct, hipIpcMemHandle_t):
        raise TypeError("Expected a hipIpcMemHandle_t instance")

    prefix = f"[Rank {rank:02}] " if rank is not None else ""

    reserved_bytes = bytes(ipc_handle_struct.reserved)
    if reserved_bytes:
        hex_values = ", ".join(f"0x{b:02x}" for b in reserved_bytes)
        print(f"{prefix}reserved_bytes: {hex_values}")
    else:
        print(f"{prefix}reserved_bytes is empty")

def initialize_and_debug(ipc_handle_data):
    # Validate and convert ipc_handle_data to bytes
    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != 64:
            raise ValueError("ipc_handle_data must be a 64-element uint8 numpy array")
        ipc_handle_data = ipc_handle_data.tobytes()
    elif isinstance(ipc_handle_data, (bytes, bytearray)):
        if len(ipc_handle_data) != 64:
            raise ValueError("ipc_handle_data must be exactly 64 bytes")
    else:
        raise TypeError("ipc_handle_data must be a numpy array, bytes, or bytearray")

    # Create hipIpcMemHandle_t instance
    raw_memory = ctypes.create_string_buffer(64)
    ipc_handle_struct = hipIpcMemHandle_t.from_buffer(raw_memory)

    # Print initial state
    print("Created hipIpcMemHandle_t instance")
    print(f"ctypes.sizeof(ipc_handle_struct): {ctypes.sizeof(ipc_handle_struct)} bytes")
    print_reserved_field(ipc_handle_struct, rank=0)

    # Copy data to ipc_handle_struct using bytes
    print("Copying data to ipc_handle_struct...")
    ctypes.memmove(ctypes.addressof(ipc_handle_struct), ipc_handle_data, 64)
    print_reserved_field(ipc_handle_struct, rank=0)


if __name__ == "__main__":
    # Test input data
    ipc_handle_data = bytearray([
        0x33, 0x35, 0x4D, 0xF8, 0xA5, 0x73, 0xE9, 0x3B, 0xB8, 0x58, 0xD9, 0xAD, 0x2F, 0xEF, 0xBC, 0xC6,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x28, 0x6B, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xD6, 0xD7, 0x25, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ])
    initialize_and_debug(ipc_handle_data)