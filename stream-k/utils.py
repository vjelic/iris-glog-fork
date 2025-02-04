import triton
import triton.language as tl

import json
import numpy as np

@triton.jit
def read_realtime():
    tmp = tl.inline_asm_elementwise(
        asm="""s_waitcnt vmcnt(0)
        s_memrealtime $0
        s_waitcnt lgkmcnt(0)""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1
    )
    return tmp

def dump_timers(mm_begin_timestamp,
                mm_end_timestamp,
                comm_begin_timestamp,
                comm_middle_max_timestamp,
                comm_middle_min_timestamp,
                comm_end_timestamp,
                gpu_freq,
                filename):
    
    cycles_to_us = lambda cycles: (cycles / gpu_freq)
    
    gemm_begin_us = cycles_to_us(mm_begin_timestamp.cpu().numpy())
    gemm_end_us = cycles_to_us(mm_end_timestamp.cpu().numpy())
    
    comm_begin_us = cycles_to_us(comm_begin_timestamp.cpu().numpy())
    poll_end_us = cycles_to_us(comm_middle_max_timestamp.cpu().numpy())
    op_begin_us = cycles_to_us(comm_middle_min_timestamp.cpu().numpy())
    op_end_us = cycles_to_us(comm_end_timestamp.cpu().numpy())


    min_timestamp = min(np.min(gemm_begin_us),
                        np.min(gemm_end_us),
                        np.min(comm_begin_us),
                        np.min(poll_end_us),
                        np.min(op_begin_us),
                        np.min(op_end_us))
    
    gemm_begin_us = gemm_begin_us - min_timestamp
    gemm_end_us = gemm_end_us - min_timestamp
    comm_begin_us = comm_begin_us - min_timestamp
    poll_end_us = poll_end_us - min_timestamp
    op_begin_us = op_begin_us - min_timestamp
    op_end_us = op_end_us - min_timestamp
    
    data = [
        {"tile_id": i,
         "gemm_begin": int(gemm_begin),
         "gemm_end":   int(gemm_end),
         "poll_begin": int(comm_begin),
         "poll_end":   int(poll_end),
         "op_begin":   int(op_begin),
         "op_end":     int(op_end,),
         "comm_begin": int(comm_begin),
         "comm_end":   int(op_end,)}
        for i, (gemm_begin, gemm_end, comm_begin,
                poll_end, op_begin, op_end) in enumerate(zip(gemm_begin_us,
                                       gemm_end_us,
                                       comm_begin_us,
                                       poll_end_us,
                                       op_begin_us,
                                       op_end_us))
    ]
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)