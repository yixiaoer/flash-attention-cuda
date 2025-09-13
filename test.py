import math

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

flashv1 = load(
    name='flashv1',
    sources=['main.cpp', 'flash_v1.cu'],
    extra_cuda_cflags=['-O2']
)

def pytorch_attn(q: Tensor, k: Tensor, v: Tensor, softmax_scale: float) -> Tensor:
    attn = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

def main():
    torch.manual_seed(42)
    B = 30
    H = 12
    N = 100  # 10, 1000, 10000
    D = 8
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    out_cu = flashv1.forward(q, k, v)
    out_pt = pytorch_attn(q, k, v, scale)

    max_diff = (out_cu - out_pt).abs().max().item()
    assert torch.allclose(out_cu, out_pt, atol=1e-5, rtol=1e-5), f'FAILED: flashv1 output differs from PyTorch (max diff: {max_diff})'
    print('✅ PASSED: flashv1 matches PyTorch manual attention.')

if __name__ == '__main__':
    main()

