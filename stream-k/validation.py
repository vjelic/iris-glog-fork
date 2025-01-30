import torch

def validate_gemm(A, B, local_C, shmem):
    expected = A @ B
    diff_mask = ~torch.isclose(local_C, expected, atol=1)
    breaking_indices = torch.nonzero(diff_mask, as_tuple=False)

    if not torch.allclose(local_C, expected, atol=1):
        max_diff = (local_C - expected).abs().max().item()
        shmem.log(f"Max absolute difference: {max_diff}")
        for idx in breaking_indices:
            idx = tuple(idx.tolist())
            local_val = local_C[idx]
            expected_val = expected[idx]
            shmem.log(f"Mismatch at index {idx}: local_C={local_val}, expected={expected_val}")
            break

    assert torch.allclose(local_C, expected, atol=1), f"max: {(local_C - expected).abs().max().item()}\n{local_C}\n{expected}"
